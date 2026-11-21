"""Playback uses frozen endpoint attributes, including clipping, for each GS."""
import numpy as np
from dataclasses import dataclass, replace
from cutsim.rendering.gs_schema import draw_from_calculation

from cutsim.rendering.clipping import CLIP_ATTRIBUTE_NAMES, NativeClipArrays, _resolve_pending_clip_draws
from cutsim.rendering.clip_fallback import append_pending_fallbacks
from cutsim.rendering.chunked_attributes import freeze_endpoint


def playback_draw_attributes(arr):
    """Expand frozen clip regions with one gather per attribute, not three."""
    names=('positions','scale','rot','coverage_logit','surface_color','surface_id')+CLIP_ATTRIBUTE_NAMES
    n=len(arr['positions'])
    joins=np.asarray(arr.get('clip_concave',np.zeros(n,bool)),dtype=bool)
    pending=np.asarray(arr.get('clip_pending',np.zeros(n,bool))) | arr.get('clip_empty',False)
    counts=1+joins.astype(np.int64)
    parents=np.repeat(np.arange(n,dtype=np.int64),counts)
    second=np.zeros(len(parents),dtype=bool)
    second[np.cumsum(counts)[joins]-1]=True
    visible=~pending[parents]
    parents,second=parents[visible],second[visible]
    out={k:arr[k][parents] for k in names if k in arr}
    if np.any(second):
        values=arr['clip_second_values'][parents[second]]
        for column,name in enumerate(CLIP_ATTRIBUTE_NAMES):
            out[name][second,0]=values[:,column]
    return append_pending_fallbacks(draw_from_calculation(out),arr)


def _select_diagnostics(mapping, rows, offset=0):
    """Remap sparse diagnostics without a Python loop over every visible GS."""
    if not mapping or not len(rows):
        return {}
    keys = np.fromiter(mapping, dtype=np.int64, count=len(mapping))
    destinations = np.searchsorted(rows, keys)  # Playback rows are ascending.
    valid = destinations < len(rows)
    keys, destinations = keys[valid], destinations[valid]
    valid = rows[destinations] == keys
    return {int(i) + offset: mapping[int(key)]
            for i, key in zip(destinations[valid], keys[valid])}


@dataclass(frozen=True)
class PreparedEndpoints:
    settings: object
    before: object
    after: object


class PlaybackAttributeCache:
    """Frozen block endpoints for one display configuration, reusable on replay."""
    def __init__(self, disk_history=False):
        self.disk_history=disk_history
        self.clear()

    def clear(self):
        self.delta = None
        self.settings = None
        self.before = self.after = None
        self._rows = self._result = None
        self._completed_endpoint = None
        self.completed_before_reused = False
        self.preparation_stats = {}
        from cutsim.core.history_store import HistoryMap
        from cutsim.core.replay_archive import ReferenceMap
        self._prepared = HistoryMap() if self.disk_history else ReferenceMap()

    @staticmethod
    def _history_key(delta):
        import uuid
        if not hasattr(delta, "_history_key"):
            delta._history_key=uuid.uuid4().hex
        return delta._history_key

    def ready(self, delta, settings):
        return self.settings == self._configuration(settings) and self._history_key(delta) in self._prepared

    @staticmethod
    def _configuration(settings):
        # The engine updates this UI hint after each cut. It is not
        # consumed by geometry, clipping or color generation (colors use levels).
        # Do not discard all frozen endpoints when the next tool changes grid h.
        return replace(settings, resolution_base_h=0.)

    def prepare_computed_block(self, delta, material, settings, cache):
        """CPU-only preparation while the compute worker exclusively owns state."""
        import os
        before_material=material.snapshot(delta.block_index)
        after_material=material.snapshot(delta.block_index+1)
        playback=delta.playback()
        fields=('positions','normals','support','tool_ids','refine_levels',
                'seam_weight','inherited_clip','sweep_domains','_local_ids')
        def arrays(cloud):
            from cutsim.core.chunked_cloud import ChunkedCloud
            if isinstance(cloud,ChunkedCloud):return cloud.chunks
            return tuple(getattr(cloud,name,None) for name in fields)
        def material_key(m):
            return (len(m.sweeps),m._sweep_index.digests[-1] if len(m.sweeps) else b'',
                    tuple(m.stock_lo),tuple(m.stock_hi),getattr(m,'_gs_density',1.),
                    tuple(getattr(m,'_surface_aliases',range(len(m.sweeps)))))
        configuration=self._configuration(settings)
        previous=self._completed_endpoint
        before=None
        if previous is not None and os.environ.get('CUTSIM_RENDER_REUSE','1')!='0':
            old_settings,old_material,old_arrays,old_attributes=previous
            current=arrays(playback.before)
            # Immutable endpoint buffers, identical row layout and append-only
            # material/provenance contract. Mere position equality is not enough.
            from cutsim.core.frozen_rows import FrozenRows
            chunks=all(isinstance(a,FrozenRows) for a in old_arrays+current)
            same=chunks and len(old_arrays)==len(current) and all(a is b for a,b in zip(old_arrays,current))
            if not chunks and not any(isinstance(a,FrozenRows) for a in old_arrays+current):
                same=len(old_arrays)==len(current) and all(a is not None and b is not None and a.shape==b.shape and a.dtype==b.dtype
                     and not a.flags.writeable and not b.flags.writeable
                     and a.strides==b.strides and (not a.size or
                     (a.ctypes.data==b.ctypes.data and np.shares_memory(a,b)))
                     for a,b in zip(old_arrays,current))
            if same and old_settings==configuration and old_material==material_key(before_material):
                before=old_attributes
        self.prepare(delta, settings, cache, before_material,after_material,_completed_before=before)
        self.completed_before_reused=before is not None
        self.preparation_stats={k:v for k,v in getattr(after_material,'last_edge_split_stats',{}).items()
                                if k.startswith('local_')}
        # Retain one endpoint only, with no geometric ancestry or delta graph.
        self._completed_endpoint=(configuration,material_key(after_material),arrays(playback.after),self.after)

    def export(self):
        """Publish completed buffers, never this cache's mutable playback cursor."""
        if self.delta is None:
            raise RuntimeError('No prepared block to publish')
        for endpoint in (self.before, self.after):
            freeze_endpoint(endpoint)
        return PreparedEndpoints(self.settings, self.before, self.after)

    def release_published(self):
        """Release archived replay payloads, retaining one warm next-block endpoint.

        Only the owning producer calls this after durable archive publication.
        Consumers keep their own packet and playback cursor.
        """
        self._prepared.clear()
        self.delta = self.before = self.after = None
        self._rows = self._result = None

    def install(self, delta, packet, settings):
        """GUI-thread adoption; a stale display configuration cannot replace it."""
        configuration = self._configuration(settings)
        if packet.settings != configuration:
            return False
        if self.settings != configuration:
            self.clear()
            self.settings = configuration
        reference = getattr(packet, '_archive_reference', None)
        if reference is not None:
            from cutsim.core.replay_archive import ReferenceMap
            if not isinstance(self._prepared, ReferenceMap):
                self._prepared = ReferenceMap()
            self._prepared.adopt(self._history_key(delta), reference.select('prepared'),
                                 (delta, packet.before, packet.after))
        else:
            self._prepared[self._history_key(delta)] = (delta, packet.before, packet.after)
        return True

    def prepare(self, delta, settings, cache, before_material, after_material, *, _completed_before=None):
        settings = self._configuration(settings)
        if self.delta is delta and self.settings == settings:
            return
        if self.settings != settings:
            self.clear()
        packet = getattr(delta, '_archive_endpoints', None)
        if packet is not None and packet.settings == settings:
            # The delta and its endpoints were decoded together. Do not load
            # the same manifest again through the attribute-history index.
            self.install(delta, packet, settings)
            self.delta, self.before, self.after = delta, packet.before, packet.after
            self._rows = self._result = None
            return
        entry = self._prepared.get(self._history_key(delta))
        if entry is not None:
            self.delta, self.before, self.after = entry
            if self.disk_history:
                for endpoint in (self.before,self.after):
                    freeze_endpoint(endpoint)
            self._rows = self._result = None
            return
        playback = delta.playback()
        # Only complete block endpoints are evaluated, never a partial sweep.
        before = (_completed_before if _completed_before is not None else
                  cache.get(playback.before, before_material, settings))
        stats=getattr(delta,'stats',{})
        empty_move=stats.get('empty_material_move',False) or stats.get('strip_shape',{}).get('empty_material_move',False)
        after = before if empty_move else cache.get(playback.after, after_material, settings)
        _resolve_pending_clip_draws(before)
        _resolve_pending_clip_draws(after)
        def row_count(value):
            if hasattr(value,'row_count'): return value.row_count
            return len(value['positions']) if isinstance(value,dict) else len(value.positions)
        if row_count(before) != row_count(playback.before) or row_count(after) != row_count(playback.after):
            raise ValueError('Playback requires one attribute record per geometric GS')
        self.delta, self.settings = delta, settings
        self.before, self.after = before, after
        self._rows = self._result = None
        value = (delta, before, after)
        from cutsim.core.replay_archive import ReferenceMap
        if isinstance(self._prepared, ReferenceMap) and hasattr(delta, '_archive_reference'):
            self._prepared.cache_value(self._history_key(delta), value)
        else:
            self._prepared[self._history_key(delta)] = value

    def get(self, u):
        if self.delta is None:
            raise RuntimeError('Prepare the completed block before playback')
        playback = self.delta.playback()
        rows = playback.rows(u)
        if rows is self._rows:
            return self._result
        if float(u) <= 0.:
            result = self.before
        elif float(u) >= 1.:
            result = self.after
        else:
            old, new = rows
            result = NativeClipArrays({name: np.concatenate((value[old], self.after[name][new]), axis=0)
                                       for name, value in self.before.items()})
            result['source_ids'] = np.arange(len(old) + len(new), dtype=np.int64)
            for name in ('pending', 'approximated'):
                previous = getattr(self.before, name, None) or {}
                final = getattr(self.after, name, None) or {}
                selected = _select_diagnostics(previous, old)
                selected.update(_select_diagnostics(final, new, len(old)))
                setattr(result, name, selected)
        self._rows, self._result = rows, result
        return result

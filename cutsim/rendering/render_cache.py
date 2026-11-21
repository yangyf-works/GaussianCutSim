"""GaussianCutSim rendering / render_cache implementation."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields, replace
import hashlib
import math
import numpy as np
import time
from cutsim.core.local_state import _LocalPointStore
from cutsim.core.material import AnalyticMaterial
from cutsim.geometry.frames import rotation_matrices_to_quaternions
from cutsim.rendering.clipping import NativeClipArrays, _native_clip_stats, _pack_native_clip
from cutsim.rendering.gaussian_attributes import EDGE_CASE_NAMES, EdgeCandidateCache, EdgeParentCache, _RenderPreparation, _edge_pair_candidates, _edge_probe_bounds, _edge_stats_empty, _edge_stats_merge, _evaluate_edge_parent, _pack_gaussian_attributes_legacy, _render_axes
from cutsim.rendering import gaussian_attributes
from cutsim.rendering.settings import RenderSettings
from cutsim.rendering.incremental_clip import peer_dirty_mask
from cutsim.core.splat_cloud import _rt_concat_clouds


class _LocalRenderArrays(dict):
    pass


class _LocalRenderCache:
    """Sweep/geometry event -> one-ring preparation -> dependency-specific fits.

    Full rebuild uses this same deterministic per-parent candidate rule. Dense
    serialization remains available for callers. Persistent CPU chunks are kept,
    but the existing GUI still uploads the combined array to its renderer.
    """
    def __init__(self):
        self.state=None; self.configuration=None; self.store=None
        self.base={}; self.legacy={}; self.candidates={}; self.output={}; self.parent_double={}
        self.legacy_double={}
        self.parts={}; self.chunks={}; self.chunk_members={}; self.parent_chunk={}
        self.edge_cache=EdgeParentCache(); self.last_result=None
        self.previous_material=None

    def _sync(self,state,settings):
        init=self.state is None or self.state.root is not state.root
        if init:
            self.__init__(); self.store=_LocalPointStore(state.root.cell_size)
            self.store.add(state.root.added); self.state=state.root
            dirty=set(self.store.points); material_dirty=set(); removed=set()
            if settings.native_two_clip:
                self._native_peer_ids={int(sid) for sid,domain in zip(state.root.added._local_ids,state.root.added.sweep_domains)
                                       if getattr(domain,'render_domains',domain) is None}
        else:
            dirty=set(); material_dirty=set(); removed=set()
            # Older saved render checkpoints predate the consumer index.
            if settings.native_two_clip and not hasattr(self,'_native_peer_ids'):
                self._native_peer_ids={sid for sid in self.store.points
                    if getattr(self.store.geometry['sweep_domains'][self.store._slots[sid]],'render_domains',
                               self.store.geometry['sweep_domains'][self.store._slots[sid]]) is None}
        self._sync_changed_clouds=[]; self._sync_born_ids=set()
        old=self.state; new=state; undo=[]; redo=[]
        while old.depth>new.depth: undo.append(old); old=old.parent
        while new.depth>old.depth: redo.append(new); new=new.parent
        while old is not new:
            undo.append(old); redo.append(new); old=old.parent; new=new.parent
        for event,reverse in [(e,True) for e in undo]+[(e,False) for e in reversed(redo)]:
            gone=event.added if reverse else event.removed
            born=event.removed if reverse else event.added
            gone_ids=getattr(gone,'_local_ids',np.empty(0,dtype=np.int64))
            born_ids=getattr(born,'_local_ids',np.empty(0,dtype=np.int64))
            if settings.native_two_clip:
                self._sync_changed_clouds.extend((gone,born))
                self._sync_born_ids.update(map(int,born_ids))
                self._native_peer_ids.difference_update(map(int,gone_ids))
                self._native_peer_ids.update(int(sid) for sid,domain in zip(born_ids,born.sweep_domains)
                                             if getattr(domain,'render_domains',domain) is None)
            else:
                dirty.update(self.store.neighbours(gone_ids,settings))
            if (settings.native_two_clip and len(gone_ids) and len(gone_ids)==len(self.store.points)
                    and all(int(sid) in self.store.points for sid in gone_ids)):
                # First Dexel publication replaces the whole display stock.
                # No surviving record needs a per-ID spatial removal here.
                self.store=_LocalPointStore(self.store.cell)
            else:self.store.remove(gone_ids)
            removed.update(map(int,gone_ids))
            self.store.add(born)
            dirty.update(map(int,born_ids))
            if not settings.native_two_clip:
                dirty.update(self.store.neighbours(born_ids,settings))
        self.state=state
        dirty={i for i in dirty if i in self.store.points}
        material_dirty={i for i in material_dirty if i in self.store.points}
        return init,dirty,material_dirty,{i for i in removed if i not in self.store.points}

    def _material_changes(self,material,settings):
        previous=self.previous_material
        self.previous_material=material.snapshot()
        if previous is None: return set()
        old=previous._sweep_index; new=material._sweep_index
        n0=len(previous.sweeps); n1=len(material.sweeps)
        low=0; high=min(n0,n1)
        # Prefix digests let appends and playback changes skip unchanged history.
        while low<high:
            mid=(low+high+1)//2
            if old.digests[mid-1]==new.digests[mid-1]: low=mid
            else: high=mid-1
        affected=set()
        for history,count in ((previous.sweeps,n0),(material.sweeps,n1)):
            for i in range(low,count): affected.update(self.store.material_dependents(history[i],settings))
        return affected

    def _prepare(self,ids,material,settings):
        # Include peers of the exact physical point, independent of dense order.
        ids=set(ids)
        for sid in tuple(ids): ids.update(self.store.groups[self.store.group_key(self.store.points[sid][0])])
        ids=np.array(sorted(ids),dtype=np.int64)
        cloud=self.store.cloud(ids); prep=_RenderPreparation(cloud)
        legacy=_pack_gaussian_attributes_legacy(cloud,material,settings,_prepared=prep)
        base=_pack_gaussian_attributes_legacy(cloud,material,replace(settings,half_double_edge_split=False),_prepared=prep)
        if settings.half_clip and settings.half_double_edge_split:
            valid,dist,dirs=_edge_pair_candidates(cloud,material,settings.sigma_scale)
        else:
            valid=np.zeros(len(ids),bool); dist=np.zeros((len(ids),2)); dirs=np.zeros((len(ids),2,3))
        lr=_local_records(legacy); br=_local_records(base)
        order=np.argsort(legacy['source_ids'],kind='stable')
        offsets=np.r_[0,np.cumsum(np.bincount(legacy['source_ids'],minlength=len(ids)))]
        changed=set()
        for row,sid0 in enumerate(ids):
            sid=int(sid0); record=br[row].copy(); record['source_ids']=sid
            candidate=(dist[row].copy(),dirs[row].copy()) if valid[row] else None
            previous=self.base.get(sid)
            old_candidate=self.candidates.get(sid)
            if previous is not None and (previous.tobytes()!=record.tobytes() or
                    (old_candidate is None)!=(candidate is None)):
                changed.add(sid)
            own=lr[order[offsets[row]:offsets[row+1]]].copy(); own['source_ids']=sid
            self.base[sid]=record; self.legacy[sid]=own; self.candidates[sid]=candidate
            self.parent_double[sid]=bool(legacy['half_double_parent'][row])
            self.legacy_double[sid]=self.parent_double[sid]
        return set(map(int,ids)),changed

    def _fit_neighbours(self,sid,settings):
        p,n,h,*_=self.store.points[sid]; h=float(h)
        radius=2.5*settings.sigma_scale*(h+self.store.max_h)+1e-10
        ids=self.store.box(p-radius,p+radius)
        ids=np.array([i for i in ids if self.candidates[int(i)] is None],dtype=np.int64)
        if not len(ids): return ids
        cloud=self.store.cloud(ids); delta=cloud.positions-p; d2=np.sum(delta*delta,axis=1)
        same=(cloud.normals@n>math.cos(math.radians(10.)))&(np.abs(delta@n)<.025*np.minimum(h,cloud.support))
        # For distance ties, sorted stable IDs have the same
        # order as retained dense rows, including in partial playback clouds.
        hit=same&(d2<(2.5*h*settings.sigma_scale+2.5*cloud.support*settings.sigma_scale)**2)
        ids=ids[hit]; d2=d2[hit]
        if len(ids)>8: ids=ids[np.argsort(d2)[:8]]
        return ids

    def _fit(self,sid,material,settings,neighbours=None):
        candidate=self.candidates[sid]
        legacy=self.legacy[sid]
        self.parent_double[sid]=self.legacy_double[sid]
        if candidate is None:
            self.parts.pop(sid,None)
            return legacy,False
        if neighbours is None: neighbours=self._fit_neighbours(sid,settings)
        ids=np.r_[sid,neighbours]; cloud=self.store.cloud(ids)
        base_records=np.array([self.base[int(i)] for i in ids],dtype=self.base[sid].dtype)
        base=_local_arrays(base_records); old=_local_arrays(legacy)
        old_ids=np.arange(len(legacy)); nn=np.arange(1,len(ids))
        distances=np.zeros((len(ids),2)); directions=np.zeros((len(ids),2,3))
        distances[0],directions[0]=candidate
        key=self.edge_cache.parent_key(cloud,0)
        dep=self.edge_cache.dependency_key(cloud,0,*candidate,old,old_ids,base,nn)
        entry,_=self.edge_cache.lookup(key,dep)
        computed=entry is None
        if entry is None:
            bu,bv=_render_axes(base['rot'])
            children,part,attempted=_evaluate_edge_parent(cloud,material,settings,0,distances,directions,
                old,base,bu,bv,old_ids,nn)
            p,n,h,*_=self.store.points[sid]
            lo,hi=_edge_probe_bounds(p,n,float(h),float(h)*settings.sigma_scale,attempted)
            self.edge_cache.put(key,dep,lo,hi,children,part)
        else:
            children,part=entry['children'],entry['stats']
        self.parts[sid]=part
        if children is None: return legacy,computed
        out=np.repeat(base_records[:1],len(children))
        for j,c in enumerate(children):
            out['positions'][j]=c['p']; out['scale'][j,:2]=c['size']
            frame=np.column_stack((c['u'],c['v'],cloud.normals[0]))
            out['rot'][j]=rotation_matrices_to_quaternions(frame[None,:,:])[0]
            alpha=float(np.clip(c['alpha'],1e-6,.999999)); op=math.log(alpha/(1-alpha))
            out['coverage_logit'][j]=op
            out['boundary_split'][j]=c['split']; out['half_active'][j]=c['half']
            out['half_double_child'][j]=c['role']!='centre'
            out['split_center_child'][j]=c['role']=='centre'; out['split_center_valid'][j]=c['role']=='centre'
            out['split_boundary_interference'][j]=c['interference']; out['split_elliptical_child'][j]=c['ellipse']
        self.parent_double[sid]=True
        return out,computed

    def get(self,cloud,material,settings,force_full=False):
        started=time.perf_counter()
        # This UI hint is not read by shading or geometry. The simulator
        # changes it with grid resolution; it must not invalidate every parent.
        configuration=(tuple((f.name,getattr(settings,f.name)) for f in fields(settings)
                             if f.name not in ('resolution_base_h','ambient','diffuse','specular')),
                       material.stock_lo.tobytes(),material.stock_hi.tobytes())
        initial,ring,material_dirty,removed=self._sync(cloud._local_state,settings)
        material_dirty.update(self._material_changes(material,settings))
        self.store.visited_points=0
        full=initial or force_full or self.configuration!=configuration
        dirty=set(self.store.points) if full else ring|material_dirty
        self.configuration=configuration
        if not dirty and not removed and self.last_result is not None:
            material.last_edge_split_stats=self.last_stats.copy()
            material.last_edge_split_stats.update(edge_recomputed=0,edge_reused=len(self.parts),
                local_updated_points=0,local_prepared_points=0,local_reference_points=0,local_changed_chunks=0,
                local_one_ring_points=0,local_material_dependents=0,local_dependency_consumers=0,
                local_removed_points=0,local_full_rebuild=0,local_reused_points=len(cloud.positions))
            material.last_edge_phase_ms={}
            return self.last_result
        changed_chunks=set()
        for sid in removed:
            key=self.parent_chunk.pop(sid,None)
            if key is not None:
                self.chunk_members[key].remove(sid); changed_chunks.add(key)
            for mapping in (self.base,self.legacy,self.candidates,self.output,self.parent_double,self.legacy_double,self.parts): mapping.pop(sid,None)
        t=time.perf_counter()
        invalidation_ms=1000*(t-started)
        if force_full: self.edge_cache.clear()
        prepared,changed=self._prepare(dirty,material,settings) if dirty else (set(),set())
        prep_ms=1000*(time.perf_counter()-t)
        t=time.perf_counter()
        # Only an ACTUAL base/candidate change can invalidate a neighbour's fit.
        # Fits read base footprints, not each other's fitted children; no recursion.
        consumers=self.store.neighbours(changed,settings) if not full else set()
        consumers={i for i in consumers if self.candidates[i] is not None}
        targets=prepared|consumers
        self.edge_cache.prepare(material,settings)
        dependencies_ms=1000*(time.perf_counter()-t)
        computed=0; refs=set(); t=time.perf_counter()
        for sid in sorted(targets):
            neighbours=self._fit_neighbours(sid,settings) if self.candidates[sid] is not None else None
            if neighbours is not None: refs.update(map(int,neighbours))
            output,did_compute=self._fit(sid,material,settings,neighbours); computed+=int(did_compute)
            self.output[sid]=output
            key=tuple(np.floor(self.store.points[sid][0]/(4*self.store.cell)).astype(np.int64))
            self.parent_chunk[sid]=key; self.chunk_members.setdefault(key,set()).add(sid); changed_chunks.add(key)
        fit_ms=1000*(time.perf_counter()-t)
        t=time.perf_counter()
        for key in changed_chunks:
            members=sorted(self.chunk_members[key])
            if members: self.chunks[key]=np.concatenate([self.output[i] for i in members])
            else:
                self.chunks.pop(key,None); self.chunk_members.pop(key,None)
        # Flat-array export and the existing GUI upload are still O(N).
        if self.chunks:
            records=np.concatenate([self.chunks[k] for k in sorted(self.chunks)])
            ids=np.asarray(cloud._local_ids,dtype=np.int64)
            owner_rows=np.searchsorted(ids,records['source_ids'])
            out=_LocalRenderArrays(_local_arrays(records)); out['source_ids']=owner_rows
            out['half_double_parent']=np.array([self.parent_double[int(i)] for i in ids],bool)
            cases={i:next(j for j,n in enumerate(EDGE_CASE_NAMES) if p[n]) for i,p in self.parts.items()}
            parent_cases=np.zeros(len(ids),np.int16)
            candidate_ids=np.fromiter(cases,dtype=np.int64,count=len(cases))
            parent_cases[np.searchsorted(ids,candidate_ids)]=np.fromiter(cases.values(),dtype=np.int16,count=len(cases))
            out['edge_case']=parent_cases[owner_rows]
        else:
            out=_LocalRenderArrays(_pack_gaussian_attributes_legacy(cloud,material,settings))
            out['edge_case']=np.zeros(0,np.int16)
        out.local_chunks=self.chunks.copy()
        total=_edge_stats_empty()
        for part in self.parts.values(): _edge_stats_merge(total,part)
        total.update(edge_candidates=len(self.parts),edge_recomputed=computed,edge_reused=len(self.parts)-computed,
            local_total_points=len(cloud.positions),local_one_ring_points=len(ring),
            local_material_dependents=len(material_dirty),local_prepared_points=len(prepared),
            local_dependency_consumers=len(consumers-prepared),local_updated_points=len(targets),
            local_reference_points=len(refs-targets),local_reused_points=len(cloud.positions)-len(targets),
            local_changed_chunks=len(changed_chunks),local_total_chunks=len(self.chunks),
            local_full_rebuild=int(full),local_removed_points=len(removed))
        material.last_edge_split_stats=total; self.last_stats=total.copy()
        material.last_edge_phase_ms=dict(local_invalidation_ms=invalidation_ms,
            local_dependencies_ms=dependencies_ms,local_prepare_ms=prep_ms,local_fit_ms=fit_ms,
            assemble_ms=1000*(time.perf_counter()-t),total_ms=1000*(time.perf_counter()-started))
        self.last_result=out
        return out


class _NativeClipCache(_LocalRenderCache):
    """Cache one record per geometric parent; concave draw expansion happens at upload."""
    def __init__(self):
        super().__init__(); self.approximated={}; self.record_store=None
        self.frozen_chunks={}

    def get(self,cloud,material,settings):
        from cutsim.core.chunked_cloud import ChunkedCloud
        chunked=isinstance(cloud,ChunkedCloud)
        count=cloud.row_count if chunked else len(cloud.positions)
        started=time.perf_counter()
        config=tuple((f.name,getattr(settings,f.name)) for f in fields(settings) if f.name not in ('resolution_base_h','ambient','diffuse','specular'))
        config+=(('gs_density',getattr(material,'_gs_density',1.)),)
        config+=(('stock_bounds',tuple(material.stock_lo),tuple(material.stock_hi)),)
        from cutsim.rendering.coverage_cuda import backend_key
        config+=(('coverage',backend_key()),)
        initial,ring,_,removed=self._sync(cloud._local_state,settings)
        dependencies=set()
        # Analytic cap ownership changes when a horizontal pass joins an
        # existing plunge, even for an otherwise unchanged geometric parent.
        from cutsim.rendering.cap_junctions import cap_junctions
        junctions=frozenset((a,b,lo,hi,tuple(p)) for a,b,lo,hi,p in cap_junctions(material))
        previous=getattr(self,'_cap_junction_signature',frozenset())
        changed_junctions=junctions.symmetric_difference(previous)
        self._cap_junction_signature=junctions
        if changed_junctions and self.record_store is not None:
            owners={owner for item in changed_junctions for owner in item[:2]}
            existing=np.array(sorted(set(self.store.points)-self._sync_born_ids),np.int64)
            if len(existing):
                labels=self.record_store['surface_id'][self.store.slots(existing),0]
                dependencies=set(map(int,existing[np.isin(labels,list(owners))]))
        full=initial or self.configuration!=config; self.configuration=config
        dirty=set(self.store.points) if full else ring|dependencies
        if not full and self._sync_changed_clouds:
            # Domain-owned rows don't read peers. Other cached rows need new
            # clipping only if a changed disk can intersect at a valid angle.
            # New/replaced parents always refresh all attributes and fallbacks.
            born={sid for sid in self._sync_born_ids if sid in self.store.points}
            # Explicit domains cannot depend on peer changes. Query the complete
            # remaining peer-consumer set directly, instead of constructing a
            # large 3-sigma neighbour union only to discard those domain rows.
            # peer_dirty_mask retains its conservative 2-sigma dependency test.
            candidates=np.array(sorted(self._native_peer_ids-born),dtype=np.int64)
            changed=_rt_concat_clouds(*self._sync_changed_clouds)
            affected=peer_dirty_mask(self.store.cloud(candidates),changed,settings)
            dirty=born | dependencies | set(map(int,candidates[affected]))
        if not dirty and not removed and self.last_result is not None:
            _native_clip_stats(self.last_result,material,local_updated_points=0,local_prepared_points=0,
                local_reused_points=count,local_total_points=count,local_full_rebuild=0)
            material.last_edge_phase_ms={}; return self.last_result
        for sid in removed:
            self.approximated.pop(sid,None)
        invalidation=1000*(time.perf_counter()-started); t=time.perf_counter()
        if dirty:
            ids=np.array(sorted(dirty),dtype=np.int64); part=_pack_native_clip(self.store.cloud(ids),material,settings,reference_store=self.store)
            if chunked:
                # Resolve once on the changed records, before their immutable
                # buckets are shared with later endpoints and the archive.
                from cutsim.rendering.clipping import _resolve_pending_clip_draws
                _resolve_pending_clip_draws(part)
            records=_local_records(part)
            slots=self.store.slots(ids)
            required=int(slots.max())+1
            if self.record_store is None or required>len(self.record_store):
                previous=self.record_store
                capacity=max(required,1024,2*len(previous) if previous is not None else 0)
                self.record_store=np.empty(capacity,dtype=records.dtype)
                if previous is not None: self.record_store[:len(previous)]=previous
            records['source_ids']=ids
            self.record_store[slots]=records
            for row,sid in enumerate(ids):
                if row in part.approximated: self.approximated[int(sid)]=part.approximated[row]
                else: self.approximated.pop(int(sid),None)
        prepare=1000*(time.perf_counter()-t); t=time.perf_counter()
        if count and chunked:
            from cutsim.core.frozen_rows import FrozenRows,BUCKET_SIZE
            from cutsim.rendering.chunked_attributes import ChunkedAttributes
            changed_buckets={int(sid)//BUCKET_SIZE for sid in dirty|set(removed)}
            geometry_chunks={int(c.ids[0])//BUCKET_SIZE:c for c in cloud.chunks if len(c.ids)}
            if not self.frozen_chunks:changed_buckets.update(geometry_chunks)
            if full:self.frozen_chunks={}
            for bucket in sorted(changed_buckets):
                geometry=geometry_chunks.get(bucket)
                if geometry is None:
                    self.frozen_chunks.pop(bucket,None);continue
                selected=geometry.ids
                records=self.record_store[self.store.slots(selected)]
                arrays={k:records[k] for k in records.dtype.names if k!='source_ids'}
                metadata={row:self.approximated[int(sid)] for row,sid in enumerate(selected) if int(sid) in self.approximated}
                self.frozen_chunks[bucket]=FrozenRows(selected,arrays,metadata)
            out=ChunkedAttributes([self.frozen_chunks[k] for k in sorted(self.frozen_chunks)],
                                  getattr(material,'_gs_density',1.))
        elif count:
            ids=np.asarray(cloud._local_ids,dtype=np.int64)
            self.frozen_chunks={}
            # Update only dirty stable-ID records. One vectorized gather owns
            # this frozen endpoint; later cache edits cannot alter old playback.
            records=self.record_store[self.store.slots(ids)]
            out=NativeClipArrays(_local_arrays(records)); out['source_ids']=np.arange(len(ids),dtype=np.int64)
            out['half_double_parent']=np.zeros(len(ids),bool)
            out.pending={}
            # Usually IDs are monotone (retained + newer births). Sparse clip
            # diagnostics need no Python dictionary over all unchanged GS.
            diagnostic_ids=np.fromiter(self.approximated,dtype=np.int64,count=len(self.approximated))
            if np.all(ids[1:]>=ids[:-1]):
                rows=np.searchsorted(ids,diagnostic_ids)
            else:
                order=np.argsort(ids,kind='stable')
                rows=order[np.searchsorted(ids[order],diagnostic_ids)]
            out.approximated={int(row):data for row,data in zip(rows,self.approximated.values())}
        else:
            self.frozen_chunks={}
            out=_pack_native_clip(cloud,material,settings)
        out.gs_density=getattr(material,'_gs_density',1.)
        out.draw_subdivision_floor_scale=settings.draw_subdivision_floor_scale
        from cutsim.rendering.clipping import _resolve_pending_clip_draws
        _resolve_pending_clip_draws(out)
        _native_clip_stats(out,material,local_updated_points=len(dirty),local_prepared_points=len(dirty),
            local_reused_points=count-len(dirty),local_total_points=count,local_one_ring_points=len(ring),
            local_material_dependents=len(dependencies),local_reference_points=0,local_full_rebuild=int(full))
        if chunked:
            material.last_edge_split_stats.update(local_changed_chunks=len(changed_buckets) if count else 0,
                                                  local_total_chunks=len(self.frozen_chunks))
        material.last_edge_phase_ms=dict(local_invalidation_ms=invalidation,local_prepare_ms=prepare,
            assemble_ms=1000*(time.perf_counter()-t),total_ms=1000*(time.perf_counter()-started))
        self.last_result=out; return out


class RenderAttributeCache:
    """Bounded GUI cache; returned arrays are read-only snapshots.

    Hash contents rather than array identities: debug stages rebuild identical
    clouds, and simulator arrays can change in place. Unknown material types
    bypass the cache because their field may depend on additional state.
    """
    _appearance_fields=('ambient','diffuse','specular','tool_id_color',
                        'resolution_color','resolution_base_h')

    def __init__(self,max_entries=3,max_bytes=96*1024*1024):
        self.max_entries=max_entries; self.max_bytes=max_bytes
        self.clear()

    def clear(self):
        self._entries=OrderedDict(); self.byte_count=0
        self.hits=0; self.misses=0; self.recolors=0
        self._edge_cache=EdgeParentCache()
        self._edge_cache.candidate_cache=EdgeCandidateCache()
        self._local_cache=_LocalRenderCache()
        self._native_cache=_NativeClipCache()

    def rebase_geometry(self, old, new):
        """Retarget an exact endpoint after history-only checkpointing.

        checkpoint_geometry copies the cloud wrapper, keeping every geometry
        array, stable ID, constraint and cell size unchanged. Only the ancestry
        changes. The point stores, dependency maps, fitted clips and last_result
        therefore remain valid. Never rebase a cache at a different endpoint.
        """
        for cache in (self._local_cache,self._native_cache):
            if cache.state is old:
                cache.state=new

    def _key(self,cloud,material,settings):
        if type(material) is not AnalyticMaterial: return None
        if (len(material.sweeps) and not material._sweep_index.supported[-1]): return None
        digest=hashlib.blake2b(digest_size=24)

        def add(value):
            if isinstance(value,np.ndarray):
                if value.dtype==object:
                    for item in value:
                        if item is None:add(None)
                        else:
                            add(len(item))
                            for region in item:add(region)
                    return
                value=np.ascontiguousarray(value)
                digest.update(repr((value.shape,value.dtype.str)).encode())
                digest.update(value.tobytes())
            else:
                digest.update(repr(value).encode())
            digest.update(b'\x00')

        for name in ('positions','normals','support','tool_ids','refine_levels','seam_weight','inherited_clip','sweep_domains'):
            add(getattr(cloud,name))
        add(material.stock_lo); add(material.stock_hi)
        add(material._sweep_index.digests[-1] if len(material.sweeps) else b'')
        for field in fields(settings):
            if field.name not in self._appearance_fields: add(getattr(settings,field.name))
        return digest.digest()

    def get(self,cloud,material,settings=RenderSettings()):
        if hasattr(cloud,'_profile_attributes'):
            from cutsim.cutting.profile_records import render_records
            from cutsim.rendering.clipping import _native_clip_stats
            self.hits+=1
            result=render_records(cloud,settings)
            _native_clip_stats(result,material,local_updated_points=0,
                local_prepared_points=0,local_reused_points=len(cloud.positions))
            material.last_edge_phase_ms={}
            return result
        if settings.native_two_clip:
            if hasattr(cloud,'_local_state') and type(material) is AnalyticMaterial:
                result=self._native_cache.get(cloud,material,settings)
                if material.last_edge_split_stats.get('local_updated_points',0): self.misses+=1
                else: self.hits+=1
                return result
            self.misses+=1
            return gaussian_attributes.pack_gaussian_attributes(cloud,material,settings)
        if hasattr(cloud,'_local_state') and type(material) is AnalyticMaterial:
            result=self._local_cache.get(cloud,material,settings)
            if material.last_edge_split_stats.get('local_updated_points',0): self.misses+=1
            else: self.hits+=1
            return result
        key=self._key(cloud,material,settings)
        appearance=tuple(getattr(settings,name) for name in self._appearance_fields)
        entry=self._entries.get(key) if key is not None else None
        if entry is not None:
            self.hits+=1
            arr,stats,previous_appearance,size=entry
            if appearance!=previous_appearance:
                # Child normals, tool IDs and refinement levels equal their
                # source parent's. Recolour without repeating topology fitting.
                shade=_pack_gaussian_attributes_legacy(cloud,material,
                    replace(settings,half_clip=False,half_double_edge_split=False))
                arr=arr.copy()
                for name in ('surface_color','colors'):
                    arr[name]=shade[name][arr['source_ids']]
                    arr[name].flags.writeable=False
                self._entries[key]=(arr,stats,appearance,size)
                self.recolors+=1
            self._entries.move_to_end(key)
            material.last_edge_phase_ms={}
            material.last_edge_split_stats=stats.copy()
            if 'edge_candidate_points' in stats:
                material.last_edge_split_stats.update(
                    edge_candidate_reused=stats['edge_candidate_points'],edge_candidate_recomputed=0,
                    edge_candidate_groups_reused=0,edge_shared_half_points=0,edge_shared_shading_points=0)
            for name in ('edge_material_aabb_pairs','edge_material_shape_excluded','edge_reused_shape_filter'):
                if name in material.last_edge_split_stats: material.last_edge_split_stats[name]=0
            if 'edge_candidates' in stats:
                material.last_edge_split_stats.update(edge_reused=stats['edge_candidates'],edge_recomputed=0)
                for reason in ('new','dependencies','material','disabled'):
                    material.last_edge_split_stats['edge_miss_'+reason]=0
            return arr
        self.misses+=1
        arr=gaussian_attributes.pack_gaussian_attributes(cloud,material,settings,edge_cache=self._edge_cache)
        size=sum(value.nbytes for value in arr.values())
        if key is not None and self.max_entries>0 and size<=self.max_bytes:
            while self._entries and (len(self._entries)>=self.max_entries or self.byte_count+size>self.max_bytes):
                _,old=self._entries.popitem(last=False)
                self.byte_count-=old[3]
            # Own the buffers so freezing a cache entry never freezes cloud data.
            arr={name:value.copy() for name,value in arr.items()}
            for value in arr.values(): value.flags.writeable=False
            stats=getattr(material,'last_edge_split_stats',{}).copy()
            self._entries[key]=(arr,stats,appearance,size)
            self.byte_count+=size
        return arr



def _local_arrays(records):
    return {k:records[k] for k in records.dtype.names}


def _local_records(arr):
    names=[k for k in arr if k!='half_double_parent']
    dtype=np.dtype([(k,arr[k].dtype,arr[k].shape[1:]) for k in names])
    out=np.empty(len(arr['positions']),dtype=dtype)
    for k in names: out[k]=arr[k]
    return out

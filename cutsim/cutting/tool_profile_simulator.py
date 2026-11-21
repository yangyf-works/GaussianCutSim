"""Transient tool-profile Dexels reconstructed from the current GS snapshot.

No cutter replay or persistent volume field. Analytic generating surfaces may
be retained alongside current GS for clipping, independently of surfaceID.
Saved deltas are playback data only; compute_next never reads an older delta.
"""
from time import perf_counter
from contextlib import ExitStack
import sys
from cutsim.rendering.gs_extent import scoped_method, support_factor, support_radius, validate_factor
import numpy as np
from cutsim.rendering.clip_search import CLIP_SIGMA
from cutsim.rendering.opaque_coverage import DEFAULT_DISPLAY_SCALE
from cutsim.core.commands import command_sweep
from cutsim.cutting.tool_profile_state import ProfileSurfaceState, combine_surfaces, RAY_STRIDE
from cutsim.cuda.profile_program import ProgramSnapshot, ProfileBlock, _snapshot
from cutsim.rendering.current_gs_clip import update_current_gs_clipping
from cutsim.rendering.gs_support_retirement import fully_cut_support_keep
from cutsim.rendering.profile_cut_clip import ProfileClip, surviving_support
from cutsim.rendering.profile_demo_gs import GaussianSnapshot

def _select(snapshot, keep):
    ep = {k: v[keep] for k, v in snapshot.endpoints.items()}
    attrs = {k: v[keep] for k, v in snapshot.gaussian.attributes.items()}
    u, v = (snapshot.gaussian.u[keep], snapshot.gaussian.v[keep])
    planes = snapshot.clipping.planes[keep]
    for a in [*ep.values(), *attrs.values(), u, v, planes]:
        a.flags.writeable = False
    stats = dict(snapshot.clipping.stats)
    sigma = attrs['scale'][:, 0]
    stats.update(clipped_gs=int(np.count_nonzero(planes[:, 2] < CLIP_SIGMA * sigma)), radius_limited_gs=int(np.count_nonzero(planes[:, 6] < CLIP_SIGMA * sigma)))
    material = snapshot.material_clipping.planes[keep]
    material.flags.writeable = False
    identity = snapshot.material_surface_ids[keep]
    identity.flags.writeable = False
    rows = np.arange(len(snapshot.endpoints['positions']))[keep]

    def raw(clip):
        source = getattr(clip, 'constraints', None)
        if hasattr(source, 'select'):
            return source.select(rows, len(snapshot.endpoints['positions']))
        return None if source is None else {i: source[int(row)] for i, row in enumerate(rows) if int(row) in source}
    result = ProgramSnapshot(ep, GaussianSnapshot(attrs, u, v, 0.0), ProfileClip(planes, stats, raw(snapshot.clipping)), ProfileClip(material, dict(stats), raw(snapshot.material_clipping)), identity, support_factor(snapshot))
    from cutsim.rendering.analytic_surfaces import select_metadata
    select_metadata(snapshot, result, keep)
    return result

class CurrentGSProgramSimulator(ProfileSurfaceState):
    material_source = 'current_gs'
    residual_threshold = 0.4
    residual_max_fragments = 1024
    adaptive_gs_scale = 1.0
    adaptive_fraction = 0.0
    adaptive_max_rays = 0
    start_replacement = False
    sweep_envelope = True
    surface_completion = True
    center_miss_material = True
    _material_cache = None
    retained_clip_repair = True
    residual_repair = True
    twoedge_first = True
    seam_replacement = True
    join_surface_ids = True
    share_ray_results = False
    edge_gap_fill = False
    clip_gpu_prepare = True
    clip_subdivision = True
    # Bounded refinement for a failed two-edge fit, including curved boundaries.
    # Unresolved/budget-limited parents keep their existing best-two result.
    clip_refinement = True
    analytic_old_clipping = True
    gs_pair_clipping = True
    clip_edge_method = 'analytic-sampled'
    adaptive = False
    gs_retirement = 'retained-or-hit'
    gs_update = 'center-ray'
    gs_pipeline = 'resident'
    # Keep old-GS replacement/cleanup; validated split children are published intact.
    stage5_enabled = True
    prune_split_children = False

    def __init__(self, *args, gs_pipeline='resident', gs_update='center-ray', gs_retirement='retained-or-hit', adaptive=False, support_sigma=1.5, clip_edge_method='analytic-sampled', capture_clipping_debug=False, clip_max_children=4096, coverage_enabled=False, coverage_repair=False, coverage_required=False, coverage_max_probes=8192, coverage_max_rays=1024, coverage_display_scale=DEFAULT_DISPLAY_SCALE, stage_recorder=None, stage5_enabled=True, addition_refinement=True, addition_max_rays=1024, **kwargs):
        if (gs_pipeline, gs_update, gs_retirement, adaptive, clip_edge_method) != ('resident', 'center-ray', 'retained-or-hit', False, 'analytic-sampled'):
            raise ValueError('Only the current CUDA center-ray / grouped analytic method is supported')
        self.support_sigma = validate_factor(support_sigma)
        self.capture_clipping_debug = bool(capture_clipping_debug)
        # Opt-in CPU copies at real stage boundaries; never used by calculation.
        self.stage_recorder = stage_recorder
        if isinstance(clip_max_children, bool) or not isinstance(clip_max_children, int) or (not 1 <= clip_max_children <= 65536):
            raise ValueError('Clipping child cap must be 1..65536')
        self.clip_max_children = clip_max_children
        # addition_max_rays bounds each GPU batch, not the total number
        # of evaluated cells. Zero retains the diagnostic base-only path.
        for name, value, low, high in (
                ('addition_max_rays', addition_max_rays, 0, 16384),
                ('coverage_max_probes', coverage_max_probes, 64, 65536),
                ('coverage_max_rays', coverage_max_rays, 0, 16384)):
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f'{name} must be an integer in [{low}, {high}]')
            setattr(self, name, value)
        for name,value in (('coverage_enabled',coverage_enabled),('coverage_repair',coverage_repair),('coverage_required',coverage_required)):
            if not isinstance(value,bool):raise ValueError(f'{name} must be a boolean')
            if value:raise ValueError(f'{name} must be False; coverage audit and repair are not supported')
        self.coverage_enabled = self.coverage_repair = self.coverage_required = False
        if not isinstance(addition_refinement, bool):
            raise ValueError('addition_refinement must be a boolean')
        self.addition_refinement = addition_refinement
        if not isinstance(stage5_enabled, bool):
            raise ValueError('stage5_enabled must be a boolean')
        self.stage5_enabled = stage5_enabled
        from cutsim.rendering.opaque_coverage import display_factor
        display_factor(coverage_display_scale)
        self.coverage_display_scale = float(coverage_display_scale)
        super().__init__(*args, **kwargs)

    @scoped_method
    def reset(self):
        super().reset()

    def make_atlas(self, block):
        atlas = super().make_atlas(block)
        from cutsim.cuda.center_ray_update import center_grid
        result = center_grid(atlas, command_sweep(self.program[block], self.stock_hi[2]), atlas.local_boxes[block], margin=self.ray_margin)
        result.profile_template = atlas
        return result

    def display_atlas(self, block):
        with self._grid_lock:
            delta = self.deltas[block] if block < len(self.deltas) else None
            stamp = (block, id(delta))
            if self._display_grid is None or getattr(self, '_adaptive_display_stamp', None) != stamp:
                atlas = self.make_atlas(block)
                if delta is not None and getattr(delta, 'adaptive', None) is not None:
                    from cutsim.cuda.adaptive_profile import append_grid, remove_parent_rays
                    atlas = remove_parent_rays(atlas, delta.adaptive.get('replaced_parents', ()))
                    atlas = append_grid(atlas, delta.adaptive['grid'])
                self._display_grid = atlas
                self._display_block = block
                self._adaptive_display_stamp = stamp
            return self._display_grid

    @scoped_method
    def initialize(self):
        super().initialize()
        if self.initial.material_clipping is None:
            self.initial.material_clipping = self.initial.clipping
        if self.initial.material_surface_ids is None:
            self.initial.material_surface_ids = self.initial.gaussian.attributes['surface_id'][:, 0]
        from cutsim.rendering.analytic_surfaces import initialize_stock
        initialize_stock(self.initial, self.stock_lo, self.stock_hi)

    @scoped_method
    def compute_next(self):
        if self.failed:
            raise RuntimeError('Reset required after a GS computation failure')
        # The guard must also cover exceptions raised by the finally block,
        # including failures after current/step_index have been committed.
        try:
            return self._compute_next()
        except BaseException:
            self.failed = True
            raise

    def _compute_next(self):
        if self.step_index >= len(self.program):
            return None
        recorder = getattr(self, 'stage_recorder', None)
        started = perf_counter()
        model = surface = None
        delta = None
        adaptive_descriptor = None
        area_candidates = None
        sweep_workspace = {}
        clipping_debug = {} if self.capture_clipping_debug else None
        try:
            self.initialize()
            block = self.step_index
            command = self.program[block]
            before = self.current
            sweep = command_sweep(command, self.stock_hi[2])
            atlas = self.make_atlas(block)
            self._active_grid = atlas
            area_stats = {}
            from cutsim.cuda.sweep_gs_selection import select_sweep_gs
            sweep_selection = select_sweep_gs(before, sweep)
            sweep_workspace['packed_cutters'] = sweep_selection.packed
            area_stats.update(sweep_selection.stats)
            from cutsim.cuda.current_gs_resident import ResidentCurrentGSSurface
            surface = ResidentCurrentGSSurface(before, self.backend, selection=sweep_selection)
            from cutsim.cuda.center_ray_update import CenterRayModel
            model = CenterRayModel(surface, atlas)
            area_candidates = None
            model.ray_margin = self.ray_margin
            self.model = model
            reconstruction = dict(model.stats)
            if recorder is not None:
                recorder.record(1, before, atlas=atlas)
            ready = perf_counter()
            group = int(self.surface_groups[block])
            from cutsim.cuda.cell_generation import short_move_sampling
            short_move, move_length, sample_pitch = short_move_sampling(command, atlas)
            use_cells = bool(self.addition_refinement and self.addition_max_rays and short_move)
            addition_descriptor = extra = None
            addition_stats = dict(addition_refinement_enabled=self.addition_refinement,
                addition_refinement_passes=0, addition_refinement_rays=0,
                addition_refinement_added=0, addition_refinement_ms=0.)
            if use_cells:
                from cutsim.cuda.cell_generation import generate_cells
                extra, addition_descriptor, addition_stats = generate_cells(model,
                    command, sweep, block, group, int(self.surface_continuity.groups[block]),
                    (self.stock_lo,self.stock_hi), batch_rays=self.addition_max_rays)
            else:
                model.step_device(sweep)
            addition_stats.update(addition_refinement_short_move=short_move,
                addition_refinement_move_length_mm=move_length,
                addition_refinement_short_threshold_mm=sample_pitch,
                addition_refinement_selection='short_move_all_cells_2x2' if use_cells else 'base_centres',
                addition_refinement_active=use_cells)
            changed = len(model.selected)
            cut = perf_counter()
            endpoint_readback_before = getattr(model, 'readback_bytes', 0)
            with surface.rt.context():
                arrays = model.endpoints_device(first_current=True)
                accepted = surface.rt.cp.ones(len(arrays['positions']), surface.rt.cp.bool_)
                selected = surface.rt.cp.flatnonzero(accepted)
                rejected = len(accepted) - len(selected)
                ep = {k: model._download(v[selected]) for k, v in arrays.items()}
                arrays = accepted = selected = None
            # All tools use centre material evidence, including a Ball ramp
            # after a plunge. No contact-only endpoints bypass that predicate.
            seam_stats = dict(seam_contact_enabled=False, seam_contact_added=0)
            read = perf_counter()
            new = _snapshot(ep, atlas, [command], surface_groups=np.array([group]))
            material_ids = new.gaussian.attributes['surface_id'][:, 0] + 3 * (int(self.surface_continuity.groups[block]) - group)
            new.material_surface_ids = material_ids
            new.material_surface_ids.flags.writeable = False
            new.endpoints['owners'] = np.full(len(ep['owners']), block, np.int64)
            new.endpoints['ray_ids'] = ep['ray_ids'] + (block + 1) * RAY_STRIDE
            new.endpoints['owners'].flags.writeable = False
            new.endpoints['ray_ids'].flags.writeable = False
            keep = np.ones(len(before.endpoints['positions']), bool)
            generated = new
            base_rays = len(atlas.origins)
            base_atlas = atlas
            total_endpoints = len(ep['positions'])
            replaced_parents = np.empty(0, np.int64)
            parent_gs_removed = 0
            adaptive_stats = dict(adaptive_enabled=self.adaptive, adaptive_budget=0, adaptive_candidates=0, adaptive_rays=0, adaptive_accepted_endpoints=0, adaptive_total_ms=0.0, adaptive_seed_ms=0.0, adaptive_cut_ms=0.0, adaptive_seed_payload_readback_bytes=0, adaptive_seed_control_readback_bytes=0, adaptive_endpoint_readback_bytes=0)
            if addition_descriptor is not None:
                from cutsim.cuda.adaptive_profile import append_grid
                generated = combine_surfaces(new, np.ones(len(ep['positions']),bool), extra)
                generated.material_surface_ids = np.r_[new.material_surface_ids,extra.material_surface_ids]
                generated.material_surface_ids.flags.writeable = False
                atlas = append_grid(atlas, addition_descriptor['grid'])
                self._active_grid = atlas
                total_endpoints += addition_stats['addition_refinement_added']
                adaptive_descriptor = addition_descriptor
            from cutsim.cuda.dexel_edge_cells import edge_generated_cells
            generated_edge_mask=edge_generated_cells(atlas,
                generated.endpoints['ray_ids']-(block+1)*RAY_STRIDE,refinement=addition_descriptor)
            sweep_workspace['new_edge_mask']=generated_edge_mask
            addition_stats['addition_edge_gs']=int(generated_edge_mask.sum())
            if recorder is not None:
                recorder.record(2, generated, model=model, stats={**model.stats,**addition_stats})
                if addition_descriptor is not None and hasattr(recorder, 'addition_refinement'):
                    recorder.addition_refinement(addition_descriptor)
            changed = len(model.selected)
            retirement_started = perf_counter()
            from cutsim.rendering.surface_replacement import replacement_candidates, commit_replacements
            replacement_mask = None
            area_stats['replacement_candidates'] = 0
            if self.stage5_enabled:
                replacement_mask = replacement_candidates(before, sweep, surface.rt, selection=sweep_selection)
                with surface.rt.context():
                    area_stats['replacement_candidates'] = int(surface.rt.cp.count_nonzero(replacement_mask).item())
            area_candidates = fully_cut_support_keep(before, sweep, 'cuda', radius_sigma=CLIP_SIGMA, device_mask=True,
                                                     selection=sweep_selection)
            from cutsim.rendering.finite_domain_retirement import finite_cut_support
            finite_removed, finite_count = finite_cut_support(before, sweep, surface.rt, area_candidates, (self.stock_lo, self.stock_hi),
                                                             selection=sweep_selection)
            with surface.rt.context():
                area_candidates |= finite_removed
                proven = int(surface.rt.cp.count_nonzero(area_candidates).item())
                ray_removed = int(surface.rt.cp.count_nonzero(model.removed).item())
                model.removed |= area_candidates
            from cutsim.rendering.analytic_residual_retirement import residual_retirement
            residual_removed, residual_metrics = residual_retirement(
                surface, sweep, model.removed, (self.stock_lo, self.stock_hi))
            with surface.rt.context():
                model.removed |= residual_removed
            area_stats.update(residual_metrics)
            from cutsim.rendering.ball_continuation_retirement import retire_ball_continuation
            area_stats.update(retire_ball_continuation(surface, sweep, model.removed,
                generated.material_surface_ids, (self.stock_lo, self.stock_hi),
                display_scale=self.coverage_display_scale))
            residual_removed = None
            area_candidates = finite_removed = None
            area_stats['finite_domain_cut_removed'] = finite_count
            area_stats.update(twoedge_retirement=True, twoedge_full_removed=proven, twoedge_ray_removed=ray_removed, retained_region_ray_candidates=ray_removed, twoedge_support_sigma=CLIP_SIGMA)
            sweep_workspace['preserve_nonempty'] = True
            sweep_workspace['retained_region'] = self.gs_retirement in ('retained-region', 'retained-or-hit')
            with surface.rt.context():
                keep = ~model._download(model.removed).astype(bool)
            seam_stats.update(seam_replacement=self.seam_replacement and self.gs_update == 'center-ray' and (self.gs_retirement not in ('full-containment', 'full-or-hit', 'residual-or-hit', 'twoedge-or-hit', 'retained-region', 'retained-or-hit')), seam_extra_rays=0, seam_extra_endpoints=0, seam_deferred=0)
            fully_cut = int(np.count_nonzero(~keep))
            retirement_ms = (perf_counter() - retirement_started) * 1000
            clip_before = before
            clip_keep = keep
            fragment_count = 0
            residual_stats = {}
            certified_ids = np.empty(0, np.int64)
            retained_fit_ids = np.empty(0, np.int64)
            prepared_old = None
            retained_stats = {}
            retained_risk = np.zeros(len(keep), bool)
            fragment_sources = np.empty(0, np.int64)
            clipping_identity = 7 + 3 * int(self.surface_continuity.groups[block])
            clipping_identity = np.unique(np.r_[clipping_identity, generated.material_surface_ids])
            after = combine_surfaces(clip_before, clip_keep, generated)
            after.material_surface_ids = np.concatenate((clip_before.material_surface_ids[clip_keep], generated.material_surface_ids))
            after.material_surface_ids.flags.writeable = False
            from cutsim.rendering.analytic_surfaces import assign_generated
            assign_generated(after, sweep, block, packed=sweep_workspace.get('packed_cutters'))
            if clipping_debug is not None or recorder is not None:
                from cutsim.diagnostics.clipping_stages import before_clipping
                before_clip = before_clipping(after, clip_before, clip_keep)
                if clipping_debug is not None:
                    clipping_debug['before_clip'] = before_clip
                if recorder is not None:
                    recorder.record(3, before_clip,
                        added=np.r_[np.zeros(int(clip_keep.sum()), bool),
                                    np.ones(len(generated.endpoints['positions']), bool)],
                        removed=~keep, stats={**area_stats, 'capture_phase': 'before_clipping'})
                del before_clip
            projection = None
            after.clipping, after.material_clipping = update_current_gs_clipping(after, clip_before, clip_keep, surface, sweep, self.backend, stock_bounds=(self.stock_lo, self.stock_hi), current_surface_id=clipping_identity, certified_fragments=fragment_count, certified_old=np.searchsorted(np.flatnonzero(keep), np.r_[certified_ids, retained_fit_ids]), prepared_old=prepared_old, collect_edges=self.clip_subdivision, gpu_prepare=self.clip_gpu_prepare, gs_pairs=self.gs_pair_clipping, projection=projection, analytic_old=self.analytic_old_clipping, sweep_workspace=sweep_workspace, analytic_pairs=self.clip_edge_method == 'analytic-pairs' and self.gs_pair_clipping, analytic_sampled=self.clip_edge_method == 'analytic-sampled' and self.gs_pair_clipping)
            retained = surviving_support(after.clipping.planes, CLIP_SIGMA * after.gaussian.attributes['scale'][:, 0])
            retained[list((after.clipping.constraints or {}).keys())] = True
            nkept = int(np.count_nonzero(keep))
            nold = nkept + fragment_count
            removed = ~keep
            empty_old = int(np.count_nonzero(~retained[:nold]))
            retained[:nold] = True
            redundant_ids = np.empty(0, np.int64)
            if len(redundant_ids):
                keep[redundant_ids] = False
                removed[redundant_ids] = True
                fully_cut = int(removed.sum())
                if 'area_retirement_removed' in area_stats:
                    area_stats['area_retirement_removed'] = fully_cut
            replacement_stats = dict(start_replacement_enabled=self.start_replacement, start_replacement_removed=0, start_replacement_ms=0.0, start_replacement_candidates=0, start_replacement_budget_skipped=0)
            added = np.r_[np.zeros(nkept, bool), np.ones(fragment_count + total_endpoints, bool)][retained]
            if not retained.all():
                after = _select(after, retained)
            # Coverage audit and supplemental generation are not pipeline stages.
            gap_stats = dict(edge_gap_added=0, edge_gap_probes=0, edge_gap_material_points=0,
                             edge_gap_analytic_unknown_points=0, edge_gap_clipping_passes=0,edge_gap_ms=0.)
            if clipping_debug is not None:
                from cutsim.diagnostics.clipping_stages import snapshot_copy
                clipping_debug['before_split'] = snapshot_copy(after)
            partition_stats = {}
            from cutsim.rendering.clip_subdivision import publish
            publication_debug = clipping_debug if clipping_debug is not None else {}
            # Only this block's stage-4/5 publications share finite cut proofs.
            # Never retain this cache in snapshots, the renderer or next block.
            trim_cache = {}
            after, added, removed, partition_stats = publish(after, added, removed, block, max_children=self.clip_max_children, gpu_prepare=self.clip_gpu_prepare, debug=publication_debug, subdivide=self.clip_refinement,trim_cache=trim_cache)
            split_children = publication_debug['split_children']
            if recorder is not None:
                recorder.record(4, after, added=added, removed=removed,
                                           stats={**after.clipping.stats, **gap_stats, **partition_stats})
            stage5_started = perf_counter()
            if self.stage5_enabled:
                replaced, replacement_detail = commit_replacements(surface, after, added, removed, replacement_mask, sweep=sweep, stock_bounds=(self.stock_lo, self.stock_hi))
                area_stats.update(replacement_detail)
                if len(replaced):
                    old_sources = np.flatnonzero(~removed)
                    old_rows = np.flatnonzero(~added)
                    removed[old_sources[np.searchsorted(old_rows, replaced)]] = True
                    published = np.ones(len(added), bool)
                    published[replaced] = False
                    after = _select(after, published)
                    added = added[published]
                    split_children = split_children[published]
                keep = ~removed
                from cutsim.rendering.residual_fragments import redundant_fragments
                old_sources = np.flatnonzero(keep)
                old_count = len(old_sources)
                fitted = np.intersect1d(certified_ids, old_sources)
                slots = np.searchsorted(old_sources, fitted)
                fragment_rows = np.flatnonzero(split_children & added)
                # Child-only cleanup is disabled by default. Old fitted GS
                # retain their existing cleanup, independently of this policy.
                residual_stats['stage5_split_children_in'] = len(fragment_rows)
                residual_stats['child_pruning_enabled'] = self.prune_split_children
                residual_stats['child_pruning_bypassed'] = 0 if self.prune_split_children else len(fragment_rows)
                candidates = np.r_[slots, fragment_rows] if self.prune_split_children else slots
                pruning_detail = {}
                if len(candidates):
                    redundant, pruning_ms = redundant_fragments(after, surface, keep, old_count, len(candidates), candidate_rows=candidates, exclude_sources=fitted, boundary_samples=True, display_scale=self.coverage_display_scale, detail=pruning_detail)
                else:
                    redundant, pruning_ms = np.zeros(0, bool), 0.
                    pruning_detail.update(residual_prune_candidates=0,residual_prune_geometry_rejected=0,
                                          residual_prune_opacity_rejected=0,residual_prune_opacity_probes=0)
                residual_stats.update(pruning_detail)
                gone = candidates[redundant]
                old_gone = gone[gone < old_count]
                removed[old_sources[old_gone]] = True
                published = np.ones(len(added), bool)
                published[gone] = False
                if len(gone):
                    after = _select(after, published)
                    added = added[published]
                residual_stats.update(twoedge_published=len(slots) - int(redundant[:len(slots)].sum()) + len(retained_fit_ids), twoedge_redundant=int(redundant[:len(slots)].sum()), residual_pruned=int(redundant[len(slots):].sum()), residual_published=len(fragment_rows) - int(redundant[len(slots):].sum()), residual_prune_ms=pruning_ms)
                residual_stats.update(stage5_split_children_removed=int(redundant[len(slots):].sum()),
                                      stage5_split_children_kept=len(fragment_rows)-int(redundant[len(slots):].sum()))
            else:
                area_stats.update(replacement_checked=0,replacement_removed=0,replacement_ms=0.)
            from cutsim.cutting.coverage_finalization import skipped_coverage
            coverage_stats, coverage_packet = skipped_coverage(scope='after_replacement_without_coverage_audit'
                if self.stage5_enabled else 'stage4_output_without_stage5')
            if recorder is not None and hasattr(recorder, 'coverage_repair'):
                recorder.coverage_repair(np.zeros(len(added), bool))
            coverage_stats['stage5_ms'] = (perf_counter()-stage5_started)*1000 if self.stage5_enabled else 0.
            trim_cache.clear()
            self._active_grid = atlas
            replacement_mask = None
            adaptive_stats.update(adaptive_enabled=adaptive_descriptor is not None,
                adaptive_budget=self.addition_max_rays if self.addition_refinement else 0,
                adaptive_candidates=0,
                adaptive_rays=addition_stats['addition_refinement_rays'],
                adaptive_accepted_endpoints=addition_stats['addition_refinement_added'])
            keep = ~removed
            fully_cut = int(removed.sum())
            source_base_rays = base_rays
            if adaptive_descriptor is not None:
                from cutsim.cuda.adaptive_profile import remove_parent_rays
                active = np.ones(len(atlas.origins), bool)
                active[replaced_parents] = False
                mapping = np.full(len(active), -1, np.int64)
                mapping[active] = np.arange(int(active.sum()))
                ids = after.endpoints['ray_ids'].copy()
                local = ids - (block + 1) * RAY_STRIDE
                current = (after.endpoints['owners'] == block) & (local >= 0) & (local < len(mapping))
                if np.any(mapping[local[current]] < 0):
                    raise AssertionError('Replaced parent GS reached publication')
                ids[current] = mapping[local[current]] + (block + 1) * RAY_STRIDE
                ids.flags.writeable = False
                if clipping_debug is not None:
                    from cutsim.diagnostics.clipping_stages import remap_stages
                    remap_stages(clipping_debug, block, mapping)
                after.endpoints = dict(after.endpoints, ray_ids=ids)
                atlas = remove_parent_rays(atlas, replaced_parents)
                self._active_grid = atlas
                base_rays -= len(replaced_parents)
                adaptive_descriptor.update(base_rays=base_rays, source_base_rays=source_base_rays, replaced_parents=replaced_parents)
            decision = self.surface_continuity.decisions[block]
            clipping = after.clipping
            if not added.any() and (not removed.any()) and np.array_equal(after.clipping.planes, before.clipping.planes) and np.array_equal(after.material_clipping.planes, before.material_clipping.planes):
                after = before
            surface.pull_stats()
            shared_stats = dict(shared_hits_enabled=False, generation_hit_links_retained=0, generation_hit_queries=0)
            ended = perf_counter()
            stats = dict(block=block + 1, backend=self.backend, material_source='current_gs', rays=len(atlas.origins), profile_rays=atlas.profile_count, clipping_source=clipping.stats.get('clipping_source', 'analytic_sweep_and_gs'), analytic_patch_count=len(getattr(after, 'analytic_patches', None) or {}), analytic_stock_plane_count=sum((p.stock for p in (getattr(after, 'analytic_patches', None) or {}).values())), analytic_group_active=clipping.stats.get('analytic_group_active', 0), analytic_group_duplicates_removed=clipping.stats.get('analytic_group_duplicates_removed', 0), analytic_continuation_candidates=clipping.stats.get('analytic_continuation_candidates', 0), analytic_continuation_refreshed=clipping.stats.get('analytic_continuation_refreshed', 0), planar_floor_domains_extended=clipping.stats.get('planar_floor_domains_extended', 0), sweep_clip_candidates=clipping.stats.get('sweep_clip_candidates', 0), sweep_clip_r_candidates_reused=clipping.stats.get('sweep_clip_r_candidates_reused', False), sweep_clip_cache_reused=clipping.stats.get('sweep_clip_cache_reused', False), sweep_clip_kernel_ms=clipping.stats.get('sweep_clip_kernel_ms', 0.0), clipping_search_sigma=clipping.stats.get('boundary_search_sigma', CLIP_SIGMA), clipping_edge_sigma=clipping.stats.get('edge_evaluation_sigma', CLIP_SIGMA), gs_pair_count=clipping.stats.get('gs_pair_count', 0), gs_pair_ridges=clipping.stats.get('gs_pair_ridges', 0), gs_pair_valleys=clipping.stats.get('gs_pair_valleys', 0), changed_rays=changed, display_gs=len(after.endpoints['positions']), initialization_ms=(ready - started) * 1000, support_sigma=self.support_sigma, cut_ms=(cut - ready) * 1000, endpoint_ms=(read - cut) * 1000, preparation_ms=(ended - read) * 1000, total_ms=(ended - started) * 1000, clipping_ms=clipping.stats['total_ms'], clipping_kernel_ms=clipping.stats['kernel_ms'], clipped_gs=clipping.stats['clipped_gs'], clipping_rejected_extrapolations=clipping.stats['rejected_extrapolated_boundaries'], clipping_finite_boundaries=clipping.stats['accepted_finite_boundaries'], clipping_boundary_retries=clipping.stats['boundary_segment_queries'], clipping_reselected_boundaries=clipping.stats['reselected_boundaries'], clipping_continuity_skips=clipping.stats.get('continuous_surface_probes_skipped', 0), clipping_curved_boundaries=clipping.stats.get('curved_boundary_planes', 0), same_surface_old_cutter_skips=clipping.stats.get('same_surface_old_cutter_skips', 0), endpoint_edge_queries=surface.stats['endpoint_edge_queries'], endpoint_edge_resolved=surface.stats['endpoint_edge_resolved'], endpoint_side_changes=surface.stats['endpoint_side_changes'], material_edge_queries=surface.stats.get('material_edge_queries', 0), material_edge_resolved=surface.stats.get('material_edge_resolved', 0), material_side_changes=surface.stats.get('material_side_changes', 0), material_confirm_queries=surface.stats.get('material_confirm_queries', 0), material_conflicts=surface.stats.get('material_conflicts', 0), radius_limited_gs=clipping.stats['radius_limited_gs'], clipping_recomputed_gs=clipping.stats['recomputed_gs'], clipping_reused_gs=clipping.stats['reused_gs'], resident_interval_rays_after_publish=0, endpoint_policy='first_current', added_gs=int(added.sum()), retained_stock_gs=int(np.count_nonzero(after.endpoints['owners'] < 0)), local_grid_bytes=sum((a.nbytes for a in vars(atlas).values() if isinstance(a, np.ndarray))), early_exit=None, surface_group=group, surface_join_reason=decision.reason, motion_kind=decision.motion, turn_degrees=decision.turn_degrees, gs_hits=reconstruction['hits'], merged_hits=reconstruction['merged_hits'], nearest_seed_rays=reconstruction['nearest_seed_rays'], inferred_crossings=reconstruction['inferred_crossings'], gs_query_points=surface.stats['query_points'], gs_query_pairs=surface.stats['query_pairs'], gs_ray_pairs=surface.stats['ray_pairs'], rejected_surface_endpoints=rejected)
            stats.update(gs_pipeline=self.gs_pipeline, base_rays=base_rays, adaptive_source_rays=source_base_rays, adaptive_replaced_parent_rays=len(replaced_parents), adaptive_removed_parent_gs=parent_gs_removed, adaptive_gs_scale=self.adaptive_gs_scale, clipping_passes=1 + gap_stats['edge_gap_clipping_passes'], clipping_stage='base_then_gap_gs' if gap_stats['edge_gap_clipping_passes'] else 'after_all_gs', adaptive_selection='foreign_surface_crossings', **adaptive_stats, fully_cut_support_gs=area_stats.get('containment_removed', fully_cut), support_retirement_ms=retirement_ms + area_stats.get('containment_ms', 0.0), **seam_stats, **replacement_stats, retired_same_surface_gs=int(np.count_nonzero(~keep & (before.material_surface_ids == 7 + 3 * int(self.surface_continuity.groups[block])))), retired_empty_support_gs=int(np.count_nonzero(~retained[nold:])), retained_empty_old_support_gs=empty_old, seed_payload_readback_bytes=reconstruction.get('seed_payload_readback_bytes'), seed_control_readback_bytes=(reconstruction.get('seed_control_readback_bytes', 0) or 0) + adaptive_stats['adaptive_seed_control_readback_bytes'], bvh_build_ms=reconstruction.get('bvh_build_ms'), gs_pack_upload_ms=reconstruction.get('gs_pack_upload_ms'), gs_to_dexel_ms=reconstruction['initialization_s'] * 1000 + adaptive_stats['adaptive_seed_ms'], endpoint_readback_bytes=getattr(model, 'readback_bytes', 0) - endpoint_readback_before + adaptive_stats['adaptive_endpoint_readback_bytes'])
            stats.update(shared_stats)
            stats['gs_update'] = self.gs_update
            stats.update({key:value for key,value in model.stats.items()
                          if key.startswith('analytic_generation_') or key in ('generation_method','generation_policy')})
            stats.update(area_stats)
            stats.update({key: model.stats.get(key, 0) for key in
                          ('boundary_support_enabled', 'boundary_support_candidates', 'boundary_support_probes', 'boundary_support_added',
                           'boundary_support_cut_rejections', 'boundary_support_sigma')})
            stats.update(gap_stats, edge_gap_fill=self.edge_gap_fill)
            stats.update(partition_stats)
            stats.update(residual_stats, residual_repair=self.residual_repair and self.gs_retirement not in ('full-containment', 'full-or-hit', 'residual-or-hit', 'twoedge-or-hit', 'retained-region', 'retained-or-hit'), surface_completion=self.surface_completion)
            stats.update(retained_stats, retained_clip_repair=self.gs_update == 'center-ray' and self.gs_retirement not in ('full-containment', 'full-or-hit', 'residual-or-hit', 'twoedge-or-hit', 'retained-region', 'retained-or-hit') and self.retained_clip_repair and self.twoedge_first and self.residual_repair)
            stats.update(gs_retirement=self.gs_retirement, sweep_envelope=self.sweep_envelope)
            stats.update(representation_retired_old=fully_cut - area_stats['twoedge_full_removed'] - area_stats['twoedge_ray_removed'] - area_stats.get('residual_removed', 0), retirement_stage='full_support_then_analytic_residual_then_geometric_replacement')
            stats.update({k: v for k, v in clipping.stats.items() if k.startswith('twoedge_') and k not in area_stats})
            stats.update({k: v for k, v in clipping.stats.items() if k.startswith('clipping_residual_')
                          or k.startswith('clipping_new_edge_') or k.startswith('clipping_truth_')
                          or k.startswith('clipping_sample_') or k.startswith('clipping_geometric_')
                          or k.startswith('clipping_unchanged_') or k.startswith('clipping_uncached_')})
            stats.update({k: v for k, v in clipping.stats.items() if k.startswith('retained_region_')})
            stats['clip_sigma'] = CLIP_SIGMA
            stats.update(center_ray_removed=fully_cut, center_ray_no_hits=model.stats['no_hit_rays'], center_miss_material=self.center_miss_material, center_hit_material_rejected=model.stats['hit_material_rejected'] + adaptive_stats.get('adaptive_hit_material_rejected', 0), center_miss_material_queries=model.stats['miss_material_candidates'] + adaptive_stats.get('adaptive_miss_material_queries', 0), center_miss_material_added=model.stats['miss_material_added'] + adaptive_stats.get('adaptive_miss_material_added', 0), center_ray_extra_intersections=0, fully_cut_support_gs=area_stats.get('containment_removed', area_stats.get('twoedge_full_removed', 0)), start_replacement_enabled=False, endpoint_policy='one_tool_surface_per_ray', center_ray_start=0.0, center_ray_payload_readback_bytes=0, center_ray_idle=not fully_cut and (not total_endpoints), early_exit=None)
            stats['adaptive_published_gs'] = int(np.count_nonzero(added & (after.endpoints['ray_ids'] - (block + 1) * RAY_STRIDE >= base_rays)))
            stats.update(coverage_stats, stage5_enabled=self.stage5_enabled,
                         adaptive_selection='short_move_all_cells_2x2' if use_cells else 'disabled')
            stats.update(addition_stats)
            stats.update(sweep_selection.stats, sweep_gs_reused_for_retirement=True)
            stats.update({k:v for k,v in clipping.stats.items() if k.startswith('analytic_face_line_')})
            local_rays=after.endpoints['ray_ids']-(block+1)*RAY_STRIDE
            stats['addition_refinement_published']=int(np.count_nonzero(added &
                (after.endpoints['owners']==block) & (local_rays>=base_rays) &
                (local_rays<base_rays+addition_stats['addition_refinement_rays'])))
            delta = ProfileBlock(block, command, sweep, before, after, removed, added, stats, adaptive_descriptor, clipping_debug)
            delta.coverage = coverage_packet
            if recorder is not None:
                recorder.record(5, after, added=added, removed=removed,
                                           stats=stats, coverage=coverage_packet)
            self.current = after
            self.deltas.append(delta)
            self.step_index += 1
            self.last_stats = stats
            return delta
        finally:
            release = perf_counter()
            area_candidates = finite_removed = None
            sweep_workspace.clear()
            replacement_mask = None
            self.model = None
            self._active_grid = None
            def release_pool():
                from cutsim.cuda.profile_dexel_cuda import runtime
                rt = runtime()
                with rt.context():
                    rt.pool.free_all_blocks()
            # ExitStack attempts every release even when an earlier one fails.
            # Register in reverse order to retain surface -> model -> pool.
            computation_error = sys.exception()
            try:
                with ExitStack() as cleanup:
                    cleanup.callback(release_pool)
                    if model is not None and hasattr(model, 'close'):
                        cleanup.callback(model.close)
                    if surface is not None:
                        cleanup.callback(surface.close)
            except BaseException as cleanup_error:
                if computation_error is not None:
                    raise cleanup_error from computation_error
                raise
            if delta is not None:
                delta.stats['release_ms'] = (perf_counter() - release) * 1000
                delta.stats['total_ms'] += delta.stats['release_ms']

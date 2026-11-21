"""CUDA Tool-profile computation adapter for the workspace/replay API."""
import numpy as np
from cutsim.core.material import AnalyticMaterial
from cutsim.core.splat_cloud import SplatCloud, _rt_snapshot_cloud
from cutsim.core.local_state import _local_initial_state, _local_take
from cutsim.core.program import default_program
from cutsim.geometry.grid_sizing import GridSizingSettings
from cutsim.geometry.sampling_density import validate_gs_density
from cutsim.cutting.block_delta import RealtimeBlockDelta
from cutsim.cutting.classification import _rt_motion_u
from cutsim.cutting.profile_records import cloud_from_snapshot
from cutsim.rendering.opaque_coverage import DEFAULT_DISPLAY_SCALE


class SweepDrivenRealtimeSimulator:
    """Expose CUDA Tool-profile computation through the workspace attributes."""

    def __init__(self, initial_splat=0.75, tool_width_cells=18,
                 min_grid_cell=0.05, max_grid_cell=1.0,
                 grid_phase=0.5, ambiguity_band_cells=0.70,
                 max_fallback_patches=28, root_iterations=20,
                 use_splat_surface_grid=False, splat_sdf_k=12,
                 adaptive_grid_refine=True,adaptive_grid_max_depth=3,
                 sweep_surface_repair=True,sweep_repair_max_new=4096,
                 shape_update_mode='strip', program=None, gs_density=1.,
                 stock_lo=(-40., -32., -12.), stock_hi=(40., 32., 0.), coverage_display_scale=DEFAULT_DISPLAY_SCALE):
        if shape_update_mode != 'strip':
            raise ValueError('Only CUDA Tool-profile Dexel is supported')
        # Workspace identifier for the CUDA Tool-profile computation adapter.
        self.shape_update_mode='strip'
        self.gs_density=validate_gs_density(gs_density)
        self.set_coverage_display_scale(coverage_display_scale)
        from cutsim.core.machining_data import stock_bounds
        stock_lo, stock_hi = stock_bounds(stock_lo, stock_hi)
        self.stock_lo=np.array(stock_lo,dtype=np.float64)
        self.stock_hi=np.array(stock_hi,dtype=np.float64)
        self.initial_splat=float(initial_splat)
        self.grid_sizing=GridSizingSettings(tool_width_cells=int(tool_width_cells),
                                             min_cell=float(min_grid_cell),
                                             max_cell=float(max_grid_cell), fixed_cell=None)
        self.grid_phase=float(grid_phase)%1.0
        self.ambiguity_band_cells=float(ambiguity_band_cells)
        self.max_fallback_patches=max(1,int(max_fallback_patches))
        self.root_iterations=int(root_iterations)
        self.min_grid_cell=float(min_grid_cell)
        # Compatibility settings for the preserved workspace controls.
        self.use_splat_surface_grid=bool(use_splat_surface_grid)
        self.splat_sdf_k=max(1,int(splat_sdf_k))
        if not isinstance(adaptive_grid_max_depth,(int,np.integer)) or not 0<=adaptive_grid_max_depth<=8:
            raise ValueError('Adaptive maximum depth must be an integer in [0,8]')
        self.adaptive_grid_refine=bool(adaptive_grid_refine)
        self.adaptive_grid_max_depth=int(adaptive_grid_max_depth)
        self.sweep_surface_repair=bool(sweep_surface_repair)
        if not isinstance(sweep_repair_max_new,(int,np.integer)) or sweep_repair_max_new<0:
            raise ValueError('Sweep repair budget must be a nonnegative integer')
        self.sweep_repair_max_new=int(sweep_repair_max_new)
        self.program=list(default_program() if program is None else program)
        self.reset()


    def set_coverage_display_scale(self, value):
        from cutsim.rendering.opaque_coverage import display_factor
        display_factor(value)
        self.coverage_display_scale = float(value)
        if getattr(self, '_engine', None) is not None:
            self._engine.coverage_display_scale = self.coverage_display_scale

    def _new_engine(self):
        from cutsim.cutting.tool_profile_simulator import CurrentGSProgramSimulator
        return CurrentGSProgramSimulator(self.program,self.stock_lo,self.stock_hi,
            gs_density=self.gs_density,initial_splat=self.initial_splat,backend='cuda',gs_pipeline='resident',
            gs_update='center-ray',gs_retirement='retained-or-hit',adaptive=False,
            support_sigma=1.5,clip_edge_method='analytic-sampled',
            coverage_display_scale=self.coverage_display_scale,stage5_enabled=True)

    def reset(self):
        self.failed=True
        self.close()
        # No CUDA handles or locks cross the multiprocessing boundary.
        engine=self._new_engine()
        self.splats=cloud_from_snapshot(engine.initial)
        self.grid_cell=engine.grid_cell
        engine.close()
        self.material=AnalyticMaterial(self.stock_lo,self.stock_hi)
        self.material._gs_density=self.gs_density
        self.step_index=0;self.deltas=[];self.last_delta=None
        self.last_sweep=None;self.last_frame=None;self.last_stats=None
        self._engine=None
        self.failed=False

    def compute_next(self):
        if self.failed:
            raise RuntimeError('Reset required after a realtime computation failure')
        try:
            return self._compute_next()
        except BaseException:
            self.failed=True
            raise

    def _compute_next(self):
        if len(self.deltas)!=self.step_index:
            raise RuntimeError('Realtime block history is out of sync; reset calculation')
        if self._engine is not None and self._engine.step_index!=self.step_index:
            raise RuntimeError('Profile and realtime block indices disagree; reset calculation')
        if self.step_index>=len(self.program):return None
        if self._engine is None:self._engine=self._new_engine()
        if self._engine.step_index!=self.step_index:
            raise RuntimeError('Profile and realtime block indices disagree; reset calculation')
        profile=self._engine.compute_next()
        if (profile is None or profile.block_index!=self.step_index
                or self._engine.step_index!=self.step_index+1):
            raise RuntimeError('Profile engine must produce exactly one ordered block; reset calculation')
        before=self.splats
        if self.step_index==0:
            # Use the exact initial endpoint published by the worker.
            before=cloud_from_snapshot(profile.before)
        after=cloud_from_snapshot(profile.after)
        removed=profile.removed_mask.copy(); added_mask=profile.added_mask
        nkept=int(np.count_nonzero(~removed))
        if (int((~added_mask).sum())!=nkept or np.any(added_mask[:nkept])
                or not np.array_equal(after.positions[:nkept],before.positions[~removed])):
            raise AssertionError('Profile endpoint ordering violates workspace playback contract')
        added=_local_take(after,np.arange(nkept,len(after.positions)))
        empty=SplatCloud.empty(); stats=dict(profile.stats)
        stats.update(total_s=stats['total_ms']/1000.,removed=int(removed.sum()),
                     added=len(added.positions),remaining=len(after.positions))
        # Populate the schema consumed by the workspace status formatter.
        stats.update(shape_update_mode='strip', removed_old=int(removed.sum()),
                     added_total=len(added.positions),
                     strip_shape=dict(changed=int(removed.sum())+len(added.positions),
                                      round_gs_intersections=True,
                                      phase_ms=dict(final_clipping=stats.get('clip_ms',0.))))
        delta=RealtimeBlockDelta(self.step_index,profile.command,profile.sweep,self.grid_cell,
            before,removed,_local_take(before,removed),empty,np.empty(0,np.int64),np.empty(0),
            _rt_motion_u(profile.sweep,before.positions),added,empty,empty,added,
            _rt_motion_u(profile.sweep,added.positions),added,
            np.ones(len(added.positions),bool),np.zeros(len(added.positions),bool),
            np.zeros(len(added.positions),bool),[],stats)
        delta._local_final_cloud=after
        delta.coverage=profile.coverage
        self.splats=after;self.step_index+=1;self.deltas.append(delta)
        self.last_delta=delta;self.last_sweep=profile.sweep;self.last_stats=stats
        # Replay is owned by the existing workspace archive, not the compute engine.
        self._engine.deltas.clear()
        return delta

    def compute_all(self):
        while self.step_index<len(self.program):yield self.compute_next()

    def close(self):
        engine=getattr(self,'_engine',None)
        if engine is not None:engine.close()
        self._engine=None

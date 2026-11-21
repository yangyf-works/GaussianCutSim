"""GaussianCutSim ui / debug_gui implementation."""
from __future__ import annotations
from cutsim.core.history_store import history_enabled

import numpy as np
import time
from cutsim.core.local_state import _local_take
from cutsim.core.splat_cloud import SplatCloud, _rt_concat_clouds, _rt_subset_cloud
from cutsim.core.sweep import BallSweep, FlatSweep
from cutsim.cutting.realtime_simulator import SweepDrivenRealtimeSimulator
from cutsim.cutting.spatial_index import _multi_edge_centres, _two_edge_centres
from cutsim.diagnostics.stats import print_rt_stats
from cutsim.rendering.clipping import _native_clip_stats, native_draw_attributes
from cutsim.rendering.clip_fallback import clip_marker_mask, size_approximation_mask
from cutsim.rendering.gaussian_attributes import _edge_split_status
from cutsim.rendering import gaussian_attributes
from cutsim.rendering.open3d_renderer import _gs_zoom_capacity, gaussian_tensor_cloud, split_draw_attributes, surface_draw_attributes, GaussianScene, require_gaussian_api
from cutsim.rendering.render_cache import RenderAttributeCache
from cutsim.rendering.playback import PlaybackAttributeCache, playback_draw_attributes
from cutsim.rendering.settings import RenderSettings
from cutsim.ui.helpers import DEBUG_LEVEL_COLORS, DEFAULT_GS_DISPLAY_SIZE, GS_DISPLAY_SIZES, display_size_index, _DisplaySizeUpdate, _SliderUpdate, _take_slider_updates, cell_boxes_lines, obb_lines, world_aabb_lines
from cutsim.ui.helpers import PLAYBACK_SPEEDS
from cutsim.ui.tool_model import ToolModelScene, tool_model_spec
from cutsim.ui.camera import update_stock_camera_clipping
from cutsim.core.commands import command_endpoints


def _block_computation_status(d):
    """Format the active algorithm's diagnostics without assuming grid stats."""
    s = d.stats
    cmd = d.command
    text = (f"Tool {cmd.tool_type} D{2 * cmd.radius:.1f} | move {d.sweep.move_length:.2f} mm | "
            f"h={d.base_h:.4f} mm\n")
    detail=s['strip_shape'];phases=detail['phase_ms']
    coverage='Coverage: disabled (not audited)\n' if s.get('coverage_status')=='disabled' else ''
    return text+("CUDA Tool-profile Dexel | current GS material\n"
        f"Changed {detail.get('changed',0):,} | removed {s['removed_old']:,} | added {s['added_total']:,}\n"
        f"Final Clipping {phases.get('final_clipping',0.):.2f} ms | TOTAL {s['total_ms']:.2f} ms\n")+coverage


class SweepDrivenRealtimeGUI:
    """UI for explaining and debugging one cutting block.

    Normal mode uses progressive birth_u/cut_u playback.
    Cut Debug mode exposes the algorithmic state transition:
        BEFORE -> REMOVE -> SPLIT -> ADD -> RESULT
    Debug overlays are geometric SplatCloud data.  They intentionally bypass the
    render-only Half2DGS split so their displayed counts match the cutting state.
    """
    DEBUG_STAGE_NAMES=('Before','Remove','Split','Add','Result')

    def __init__(self,sim: SweepDrivenRealtimeSimulator):
        import open3d as o3d
        from open3d.visualization import gui,rendering
        require_gaussian_api(o3d)
        self.o3d,self.gui,self.rendering=o3d,gui,rendering
        self.sim=sim
        self.render_delta_index=-1
        self.progress=0.0
        self.debug_stage='Before'
        self._playing=False
        self._playback_time=None
        self._playback_next_frame=None
        self._playback_prepare_index=None
        self._playback_settings=None
        self._half_material_cache={}
        self._render_attribute_cache=RenderAttributeCache()
        self._playback_attributes=PlaybackAttributeCache(disk_history=history_enabled())
        self._sigma_updates=_DisplaySizeUpdate(DEFAULT_GS_DISPLAY_SIZE)
        self._progress_updates=_SliderUpdate(0.0,0.0,1.0)
        self._speed_updates=_SliderUpdate(1.0,PLAYBACK_SPEEDS[0],PLAYBACK_SPEEDS[-1])
        self._syncing_progress=False
        self._viewport_size=None
        self._viewport_pending_at=None

        gui.Application.instance.initialize()
        self.window=self._create_window('GaussianCutSim',1680,960)
        self.window.set_on_tick_event(self._on_view_tick)
        self.scene=gui.SceneWidget(); self.scene.scene=rendering.Open3DScene(self.window.renderer)
        from cutsim.ui.mouse_controls import camera_mouse_handler
        self.scene.set_on_mouse(camera_mouse_handler(gui))
        # force_redraw() only invalidates the 3D view when scene caching is on.
        # Redraw requests must work without mouse events over the viewport.
        self.scene.enable_scene_caching(True)
        self.scene.scene.set_background(np.array([0.055,0.065,0.08,1]))
        self.scene.scene.set_lighting(rendering.Open3DScene.LightingProfile.NO_SHADOWS,
                                      np.array([-.45,-.55,-.70],dtype=np.float32))
        self.scene.scene.view.set_shadowing(False)
        self.panel=self._create_debug_panel()
        self.window.add_child(self.scene); self.window.add_child(self.panel)
        self.window.set_on_layout(self._layout)

        panel=self._debug_panel_parent('overview')
        panel.add_child(gui.Label('2DGS CUTTING — ALGORITHM DEBUG'))
        profile=getattr(sim,'shape_update_mode','')=='profile_dexel'
        description='CUDA Tool-profile Dexel: current clipped GS + current tool -> local intervals -> 2DGS'
        panel.add_child(gui.Label(description+'\n'+' -> '.join(self.DEBUG_STAGE_NAMES)))

        panel=self._debug_panel_parent('playback')
        # ---- CUT ---------------------------------------------------------
        if not getattr(self,'HEADER_COMPUTE',False):panel.add_child(gui.Label('=== CUT ==='))
        row=gui.Horiz(5)
        b=gui.Button('Compute All'); b.set_on_clicked(self._compute_all); row.add_child(b)
        b=gui.Button('Reset'); b.set_on_clicked(self._reset); row.add_child(b)
        panel.add_child(row)
        row.visible=not getattr(self,'HEADER_COMPUTE',False)

        row=gui.Horiz(5)
        b=gui.Button('< Block'); b.set_on_clicked(self._prev_buffer); row.add_child(b)
        b=gui.Button('Block >'); b.set_on_clicked(self._next_buffer); row.add_child(b)
        b=gui.Button('Play'); b.set_on_clicked(self._play); row.add_child(b)
        b=gui.Button('Stop'); b.set_on_clicked(self._stop); row.add_child(b)
        panel.add_child(row)

        prow=gui.Horiz(5); prow.add_child(gui.Label('Motion u'))
        self.progress_slider=gui.Slider(gui.Slider.DOUBLE); self.progress_slider.set_limits(0.0,100.0)
        self.progress_slider.double_value=0.0; self.progress_slider.set_on_value_changed(self._progress_changed)
        prow.add_child(self.progress_slider)
        self.progress_label=gui.Label('0.00'); prow.add_child(self.progress_label); panel.add_child(prow)

        srow=gui.Horiz(5); srow.add_child(gui.Label('Playback speed'))
        self.speed_slider=gui.Slider(gui.Slider.DOUBLE); self.speed_slider.set_limits(PLAYBACK_SPEEDS[0],PLAYBACK_SPEEDS[-1])
        self.speed_slider.double_value=1.0
        self.speed_slider.set_on_value_changed(self._speed_changed)
        srow.add_child(self.speed_slider)
        self.speed_label=gui.Label('1.00x'); srow.add_child(self.speed_label); panel.add_child(srow)
        srow.visible=getattr(self,'SHOW_PLAYBACK_SPEED',True)

        panel=self._debug_panel_parent('stages')
        # ---- ALGORITHM DEBUG --------------------------------------------
        if not getattr(self,'HEADER_COMPUTE',False):panel.add_child(gui.Label('=== CUT DEBUG / ALGORITHM STEPS ==='))
        self.cut_debug=gui.Checkbox('Cut Debug (show one block as algorithm stages)')
        self.cut_debug.checked=True; self.cut_debug.set_on_checked(lambda _x:self._redraw())
        panel.add_child(self.cut_debug)

        stage_row=gui.Horiz(4)
        for label in self.DEBUG_STAGE_NAMES:
            b=gui.Button(label)
            b.set_on_clicked(lambda _x=None,stage=label:self._set_debug_stage(stage))
            stage_row.add_child(b)
        panel.add_child(stage_row)
        self.stage_label=gui.Label('Stage: Before')
        panel.add_child(self.stage_label)

        self.show_removed=gui.Checkbox('Retired Dexel boundary samples [RED]' if profile else 'Removed parent 2DGS  [RED]')
        self.show_removed.checked=False; self.show_removed.set_on_checked(lambda _x:self._redraw()); panel.add_child(self.show_removed)
        self.show_split=gui.Checkbox('Surviving geometric split children  [ORANGE]')
        self.show_split.checked=False; self.show_split.set_on_checked(lambda _x:self._redraw()); panel.add_child(self.show_split)
        self.show_added_direct=gui.Checkbox('New Dexel boundary samples [GREEN]' if profile else 'Added from SweepSurface  [GREEN]')
        self.show_added_direct.checked=False; self.show_added_direct.set_on_checked(lambda _x:self._redraw()); panel.add_child(self.show_added_direct)
        self.show_added_grid=gui.Checkbox('Added from Local Grid  [CYAN]')
        self.show_added_grid.checked=False; self.show_added_grid.set_on_checked(lambda _x:self._redraw()); panel.add_child(self.show_added_grid)
        self.show_vertex_refine=gui.Checkbox('3-face Grid trigger candidates  [YELLOW POINTS]')
        self.show_vertex_refine.checked=False; self.show_vertex_refine.set_on_checked(lambda _x:self._redraw()); panel.add_child(self.show_vertex_refine)
        if not profile:panel.add_child(gui.Label('Yellow points mark Grid requests, not added surface.\nCandidate positions can be in air; rebuilt surface is CYAN.'))
        self.show_final=gui.Checkbox('Final surface as context')
        self.show_final.checked=False; self.show_final.set_on_checked(lambda _x:self._redraw()); panel.add_child(self.show_final)
        panel.add_child(gui.Label(
            'Before / Remove / Add / Result compare boundary samples.\nRed and green are sample changes, not removed/added solid volume.'
            if profile else
            'Geometric Split = surviving OLD surface after partial cutting.\n2-edge Half2DGS below remains a separate render-only operation.'
            if 'Split' in self.DEBUG_STAGE_NAMES else
            'Geometric Split is disabled. Partial GS retain their original frame.\nDrawing-only Clipping subdivision is a separate operation.'))
        self.debug_counts=gui.Label('Removed 0 | Split 0 | SweepSurface +0 | LocalGrid +0')
        panel.add_child(self.debug_counts)

        panel=self._debug_panel_parent('display')
        # ---- DISPLAY ----------------------------------------------------
        if not getattr(self,'HEADER_COMPUTE',False):panel.add_child(gui.Label('=== DISPLAY ==='))
        save_case=gui.Button('Save fixed GS + camera')
        save_case.set_on_clicked(self._save_render_case)
        panel.add_child(save_case)
        self._add_draw_subdivision_controls()
        sigrow=gui.Horiz(5); sigrow.add_child(gui.Label('2DGS Display Size'))
        self.sigma=gui.Combobox()
        for value in GS_DISPLAY_SIZES:self.sigma.add_item(f'{value:.2f}x')
        self.sigma.selected_index=display_size_index(self._sigma_updates.applied)
        self.sigma.set_on_selection_changed(lambda _label,index:self._display_size_changed(GS_DISPLAY_SIZES[index]))
        sigrow.add_child(self.sigma); panel.add_child(sigrow)
        self.opaque_surface=gui.Checkbox('Opaque surface smoothing')
        self.opaque_surface.checked=True
        self.opaque_surface.set_on_checked(self._opaque_surface_changed)
        panel.add_child(self.opaque_surface)
        self.show_gs=gui.Checkbox('Show GS (fixed centre alpha 0.99)')
        self.show_gs.checked=True
        self.show_gs.set_on_checked(self._gs_visibility_changed)
        panel.add_child(self.show_gs)
        self.lighting_controls=[]
        for index, (label, value) in enumerate(zip(('Ambient', 'Diffuse', 'Specular'), (.5, .5, .5))):
            row=gui.Horiz(5); row.add_child(gui.Label(label))
            control=gui.NumberEdit(gui.NumberEdit.DOUBLE)
            control.double_value=value
            control.set_on_value_changed(lambda value, i=index: self._lighting_changed(i, value))
            self.lighting_controls.append(control); row.add_child(control); panel.add_child(row)
        row=gui.Horiz(5); row.add_child(gui.Label('Shininess'))
        self.shininess_control=gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.shininess_control.double_value=64.
        self.shininess_control.set_limits(1., 256.)
        self.shininess_control.set_on_value_changed(self._shininess_changed)
        row.add_child(self.shininess_control); panel.add_child(row)
        panel.add_child(gui.Label('Render only — geometric support and cutting decisions are unchanged.'))

        panel.add_child(gui.Label('Analytic clipping: up to 2 edges per GS; validated child subdivision when needed.\nStock and cut-region constraints are retained.'))
        self.half_gs=gui.Checkbox('Boundary clipping')
        self.half_gs.checked=True; self.half_gs.set_on_checked(lambda _x:self._redraw()); panel.add_child(self.half_gs)
        self.show_two_edge=gui.Checkbox('2 applied edge centres  [CYAN]')
        self.show_two_edge.checked=False; self.show_two_edge.set_on_checked(lambda _x:self._redraw()); panel.add_child(self.show_two_edge)
        if not profile:panel.add_child(gui.Label('Marks parents with 2 selected edges, including Top-2.\nIf both markers are ON, 3+ candidates use MAGENTA.'))
        self.show_multi_edge=gui.Checkbox('3+ boundary centres  [MAGENTA]')
        self.show_multi_edge.checked=False; self.show_multi_edge.set_on_checked(lambda _x:self._redraw()); panel.add_child(self.show_multi_edge)
        if not profile:panel.add_child(gui.Label('Marks 3+ geometric ridge boundaries before selection.\nUnsupported clip domains use render subdivision; unresolved remainders stay MAGENTA.'))
        self.tool_colors=gui.Checkbox('Color by cutting tool')
        self.tool_colors.checked=True; self.tool_colors.set_on_checked(lambda _x:self._redraw()); panel.add_child(self.tool_colors)
        self.show_tool=gui.Checkbox('Show tool model')
        self.show_tool.checked=False; self.show_tool.set_on_checked(self._set_tool_visible); panel.add_child(self.show_tool)
        self.res_colors=gui.Checkbox('Color by resolution')
        self.res_colors.checked=False; self.res_colors.set_on_checked(lambda _x:self._redraw()); panel.add_child(self.res_colors)

        # ---- GRID DEBUG -------------------------------------------------
        panel=self._debug_panel_parent('grid')
        panel.add_child(gui.Label('=== PROFILE GRID DEBUG ===' if profile else '=== GRID / CLASSIFICATION DEBUG ==='))
        self.show_candidates=gui.Checkbox('Candidates: Direct / Grid / Discard')
        self.show_candidates.checked=False; self.show_candidates.set_on_checked(lambda _x:self._redraw()); panel.add_child(self.show_candidates)
        self.show_grid=gui.Checkbox('Show local Dexels' if profile else 'Adaptive fallback grids + crossing cells')
        self.show_grid.checked=False; self.show_grid.set_on_checked(lambda _x:self._redraw())
        if profile and hasattr(self,'profile_grid_controls'):
            self.profile_grid_controls.add_child(self.show_grid)
            self.profile_grid_controls.add_child(self.all_local_dexels)
        else:panel.add_child(self.show_grid)
        self.show_bounds=gui.Checkbox('Newest sweep AABB')
        self.show_bounds.checked=False; self.show_bounds.set_on_checked(lambda _x:self._redraw()); panel.add_child(self.show_bounds)

        panel=self._debug_panel_parent('status')
        if not getattr(self,'HEADER_COMPUTE',False):panel.add_child(gui.Label('=== STATUS ==='))
        self.status=gui.Label(''); panel.add_child(self.status)

        self.gs_mat=rendering.MaterialRecord(); self.has_gs=True
        self.gs_mat.gaussian_splat_max_tiles_per_splat=1024
        self._gs_scene=GaussianScene(self.scene.scene, (.5,.5,.5), self.opaque_surface.checked)
        self.point_mat=rendering.MaterialRecord(); self.point_mat.shader='defaultUnlit'; self.point_mat.point_size=3.0
        self.candidate_mat=rendering.MaterialRecord(); self.candidate_mat.shader='defaultUnlit'; self.candidate_mat.point_size=5.0
        self.line_mat=rendering.MaterialRecord(); self.line_mat.shader='unlitLine'; self.line_mat.line_width=1.5
        self._tool_scene=ToolModelScene(o3d,self.scene.scene)
        self._debug_names=[]

        self._redraw()
        camera_scale=float(np.max(sim.stock_hi-sim.stock_lo))/80.
        bounds=o3d.geometry.AxisAlignedBoundingBox(sim.stock_lo-np.array([5.,5.,2.])*camera_scale,
                                                 sim.stock_hi+np.array([5.,5.,12.])*camera_scale)
        center=(sim.stock_lo+sim.stock_hi)*.5+np.array([0.,0.,3.])*camera_scale
        self.scene.setup_camera(35,bounds,center)
        self.scene.look_at(center,center+np.array([90.,-110.,75.])*camera_scale,np.array([0.,0.,1.]))
        self._tool_scene.light.update()

    def _add_draw_subdivision_controls(self):
        """Optional drawing-quality controls supplied by the strip debugger."""

    def _debug_panel_parent(self,section):
        return self.panel

    def _create_debug_panel(self):
        return self.gui.Vert(6,self.gui.Margins(10,10,10,10))

    def _layout(self,_):
        r=self.window.content_rect
        w=min(620,max(520,int(r.width*0.37)))
        self.scene.frame=self.gui.Rect(r.x,r.y,max(320,r.width-w),r.height)
        self.panel.frame=self.gui.Rect(r.get_right()-w,r.y,w,r.height)
        size=(self.scene.frame.width,self.scene.frame.height)
        if size!=self._viewport_size:
            self._viewport_size=size; self._viewport_pending_at=time.perf_counter()

    def _configure_gs_capacity(self,count):
        frame=self.scene.frame; r=self.window.content_rect
        width=frame.width or max(320,r.width-min(620,max(520,int(r.width*.37))))
        height=frame.height or max(1,r.height)
        cap,average=_gs_zoom_capacity(width,height,count)
        self.gs_mat.gaussian_splat_max_tiles_per_splat=cap
        self.gs_mat.gaussian_splat_avg_tiles_per_splat=average
        self.gs_mat.gaussian_splat_max_tile_entries_total=32*1024*1024

    def _remove(self,name):
        self._gs_scene.entries.pop(name, None)
        if self.scene.scene.has_geometry(name): self.scene.scene.remove_geometry(name)

    def _render_settings(self, half_clip=False):
        return RenderSettings(native_two_clip=True,
                              sigma_scale=self._sigma_updates.applied,alpha=0.99,
                              ambient=0.50,diffuse=0.50,specular=0.50,
                              half_clip=bool(half_clip),
                              tool_id_color=bool(self.tool_colors.checked),
                              resolution_color=bool(self.res_colors.checked),
                              resolution_base_h=float(self.sim.grid_cell),
                              resolution_seam_scale=1.20,
                              half_min_angle_deg=5.0,
                              half_double_edge_split=False,
                              half_double_edge_elliptical=False,
                              half_double_edge_offset_scale=0.34,
                              half_double_edge_min_sigma_scale=0.78,
                              half_double_edge_opacity_scale=0.72)

    def _debug_render_settings(self):
        # Debug layers show the actual geometric SplatCloud entries.  Half clipping
        # and resolution-boundary expansion are disabled so counts map 1:1 to geometry.
        return RenderSettings(sigma_scale=self._sigma_updates.applied,alpha=0.99,
                              ambient=0.50,diffuse=0.50,specular=0.50,
                              half_clip=False,tool_id_color=False,resolution_color=False,
                              resolution_base_h=float(self.sim.grid_cell),
                              resolution_seam_scale=1.0,half_double_edge_split=False)

    def _current_delta(self):
        if 0 <= self.render_delta_index < len(self.sim.deltas):
            return self.sim.deltas[self.render_delta_index]
        return None

    @staticmethod
    def _retained_cloud(d):
        count=int(np.count_nonzero(~d.removed_mask))
        if hasattr(d,'_local_final_cloud'):return _local_take(d._local_final_cloud,np.arange(count))
        return _rt_subset_cloud(d.before,~d.removed_mask)

    @staticmethod
    def _final_cloud(d):
        if hasattr(d,'_local_final_cloud'): return d._local_final_cloud
        return _rt_concat_clouds(_rt_subset_cloud(d.before,~d.removed_mask),d.split_children,d.added)

    def _stage_base_cloud(self,d):
        if d is None:
            return self.sim.splats if not self.sim.deltas else self.sim.deltas[0].before
        stage=self.debug_stage
        if stage=='Before':
            return d.before
        retained=self._retained_cloud(d)
        if stage=='Remove':
            return retained
        if stage in ('Split','Add'):
            return _rt_concat_clouds(retained,d.split_children)
        return self._final_cloud(d)

    def _visible_cloud(self):
        d=self._current_delta()
        if d is None:
            return self.sim.splats if not self.sim.deltas else self.sim.deltas[0].before
        if self.cut_debug.checked:
            return self._stage_base_cloud(d)
        return d.visible_cloud(self.progress)

    def _material_snapshot(self, sweep_count: int):
        sweep_count=max(0,min(int(sweep_count),len(self.sim.material.sweeps)))
        m=self._half_material_cache.pop(sweep_count,None)
        if m is None:
            m=self.sim.material.snapshot(sweep_count)
        self._half_material_cache[sweep_count]=m
        # The complete analytic history remains in sim.material. Keeping every
        # old GUI context would pin separate transported provenance dictionaries
        # while Compute all and Play advance together.
        while len(self._half_material_cache)>4:
            self._half_material_cache.pop(next(iter(self._half_material_cache)))
        return m

    def _material_for_progress(self, d, u: float):
        if d is None: return self._material_snapshot(0)
        # Algorithm-stage debugging shows completed block geometry for all stages
        # except Before.  This keeps Half2DGS context deterministic while inspecting.
        if self.cut_debug.checked:
            return self._material_snapshot(d.block_index if self.debug_stage=='Before' else d.block_index+1)
        # Playback uploads endpoint attributes; this material is only block context.
        return self._material_snapshot(d.block_index if u <= 0. else d.block_index+1)

    def _set_render_delta(self,index,u=0.0):
        self._stop()
        if not self.sim.deltas:
            self.render_delta_index=-1; self.progress=0.0
        else:
            self.render_delta_index=int(np.clip(index,0,len(self.sim.deltas)-1)); self.progress=float(np.clip(u,0,1))
        self._sync_progress(self.progress)
        self._redraw()

    def _compute_next(self):
        d=self.sim.compute_next()
        if d is not None:
            print_rt_stats(d.stats)
            self._set_render_delta(len(self.sim.deltas)-1,0.0)
        else: self._redraw()

    def _compute_all(self):
        start=len(self.sim.deltas)
        for d in self.sim.compute_all(): print_rt_stats(d.stats)
        if len(self.sim.deltas): self._set_render_delta(start if start<len(self.sim.deltas) else len(self.sim.deltas)-1,0.0)
        else: self._redraw()

    def _prev_buffer(self): self._set_render_delta(self.render_delta_index-1,0.0)
    def _next_buffer(self): self._set_render_delta(self.render_delta_index+1,0.0)

    def _sync_progress(self,value):
        # Playback/navigation owns this value: discard any obsolete manual seek.
        self.progress=self._progress_updates.reset(value)
        self._syncing_progress=True
        try:
            # Open3D's visible slider round-trips through float32 during Draw.
            # Supplying an arbitrary double triggers a deferred value-changed
            # callback after _syncing_progress has already been cleared.
            self.progress_slider.double_value=float(np.float32(100.0*self.progress))
        finally:
            self._syncing_progress=False
        self.progress_label.text=f'{self.progress:.2f}'

    def _progress_changed(self,value):
        if self._syncing_progress: return
        if float(np.float32(value))==float(np.float32(100.0*self.progress)):
            return  # Widget echo of our displayed value, not a manual seek.
        self._stop()  # Old playback frames must never overwrite a manual drag.
        value=self._progress_updates.request(float(value)/100.0)
        self.progress_label.text=f'{value:.2f}'

    def _speed_changed(self,value):
        # Speed changes only the motion clock; no expensive geometry rebuild
        # needs the debounce used for display-size and progress edits.
        value=self._speed_updates.reset(value)
        self.speed_label.text=f'{value:.2f}x'

    def _display_size_changed(self,value):
        index=display_size_index(value)
        self._sigma_updates.request(GS_DISPLAY_SIZES[index])
        if self.sigma.selected_index!=index:self.sigma.selected_index=index
        if hasattr(self,'workspace_size') and self.workspace_size.selected_index!=index:
            self.workspace_size.selected_index=index

    def _gs_visibility_changed(self, value):
        self._gs_scene.set_visible(value)
        self._request_redraw()

    def _request_redraw(self):
        """Invalidate the active cached 3D view as well as the window."""
        widget = self._tool_preview.widget if getattr(self, '_tool_preview_active', False) else self.scene
        widget.force_redraw()
        self.window.post_redraw()

    def _lighting_changed(self, index, value):
        lighting=self._gs_scene.lighting.copy(); lighting[index]=value
        try:
            self._gs_scene.set_style(lighting=lighting)
        except ValueError as exc:
            self.status.text=str(exc)
        self.lighting_controls[index].double_value=float(self._gs_scene.lighting[index])
        self._request_redraw()

    def _shininess_changed(self, value):
        try:
            self._gs_scene.set_style(shininess=value)
        except ValueError as exc:
            self.status.text=str(exc)
        self.shininess_control.double_value=self._gs_scene.shininess
        self._request_redraw()

    def _opaque_surface_changed(self,value):
        self._gs_scene.set_style(opaque=value)
        self._request_redraw()

    def _prepare_surface_draw(self,draw,settings,solid_color):
        return surface_draw_attributes(draw,solid_color)

    def _on_view_tick(self):
        # Open3D's mouse controls retain near while dollying and may update far.
        # Follow the actual camera before rendering, also when paused, computing
        # or waiting for the first playable block. No geometry is regenerated.
        camera_changed = False
        if getattr(self, '_tool_preview_active', False):
            light_changed = self._tool_preview.light.update()
        else:
            frame = self.scene.frame
            camera_changed = update_stock_camera_clipping(
                self.scene.scene.camera, self.sim.stock_lo, self.sim.stock_hi,
                frame.width, frame.height)
            light_changed = self._tool_scene.light.update()
        if camera_changed or light_changed:
            self._request_redraw()
        return self._apply_slider_updates() or camera_changed or light_changed

    def _apply_slider_updates(self):
        sigma, progress, speed = _take_slider_updates(
            self._sigma_updates,self._progress_updates,self._speed_updates)
        if progress is not None: self.progress=progress
        if self._playing:
            settings=self._render_settings(half_clip=bool(self.half_gs.checked))
            if settings!=self._playback_settings:
                self._playback_settings=settings
                self._playback_prepare_index=self.render_delta_index
                self._playback_time=self._playback_next_frame=None
            if self._playback_prepare_index is not None:
                try:
                    self._prepare_playback()
                except Exception:
                    self._stop()
                    raise
                return True
        frame_applied=self._advance_playback()
        if (frame_applied and self._playing and
                not self._playback_attributes.ready(self.sim.deltas[self.render_delta_index],self._playback_settings)):
            # A display change or standalone Debug computation can leave a later
            # block cold. Show preparation status and pause the motion clock;
            # never include preparation time in tool motion or wait for all blocks.
            self._playback_prepare_index=self.render_delta_index
            self._playback_time=self._playback_next_frame=None
            self.status.text=f'Preparing 30 fps playback: block {self.render_delta_index+1}/{len(self.sim.deltas)}'
            return True
        pending=getattr(self,'_viewport_pending_at',None)
        resizing=(pending is not None and time.perf_counter()-pending>=.30 and
                  all(u.pending is None for u in (self._sigma_updates,self._progress_updates,self._speed_updates)))
        if resizing:self._viewport_pending_at=None
        if sigma is not None or progress is not None or frame_applied or resizing:
            try:
                self._redraw()
            except Exception:
                self._stop()
                raise
            # Preparation of a new block/display configuration is not motion.
            if self._playing and (self._playback_time is None or sigma is not None):
                self._playback_time=time.perf_counter()
                self._playback_next_frame=self._playback_time+1.0/30.0
            return True
        # Playback reads the committed scalar; speed alone does not rebuild GS.
        return speed is not None

    def _set_debug_stage(self,stage):
        self._stop(); self._sync_progress(self.progress)
        self.debug_stage=str(stage)
        self.cut_debug.checked=True
        # Stage buttons are useful presets; all layer checkboxes remain independently editable afterwards.
        self.show_removed.checked=(stage=='Remove')
        self.show_split.checked=(stage=='Split')
        self.show_added_direct.checked=(stage=='Add')
        self.show_added_grid.checked=(stage=='Add')
        # Trigger candidates are diagnostics, not members of the Added surface.
        self.show_vertex_refine.checked=False
        self.show_final.checked=False
        self.stage_label.text=f'Stage: {stage}'
        self._redraw()

    def _stop(self):
        self._playing=False
        self._playback_time=None
        self._playback_next_frame=None
        self._playback_prepare_index=None

    def _prepare_playback(self):
        """Prepare only the current block, never gate startup on future blocks."""
        settings=self._playback_settings
        di=self.render_delta_index
        d=self.sim.deltas[di]
        if not self._playback_attributes.ready(d,settings):
            self.status.text=f'Preparing 30 fps playback: block {di+1}/{len(self.sim.deltas)}'
            self._playback_attributes.prepare(d,settings,self._render_attribute_cache,
                self._material_snapshot(d.block_index),self._material_snapshot(d.block_index+1))
        self._playback_prepare_index=None
        self._redraw()
        self._playback_time=time.perf_counter()
        self._playback_next_frame=self._playback_time+1.0/30.0

    def _advance_playback(self):
        """Advance on the GUI tick; no posted frames or worker acknowledgements."""
        if not self._playing: return False
        if not self.sim.deltas:
            self._stop()
            return False
        now=time.perf_counter()
        if self._playback_next_frame is not None and now+1e-9<self._playback_next_frame:
            return False
        if self.progress>=1.0:
            if self.render_delta_index+1>=len(self.sim.deltas):
                if self._playback_has_pending_blocks():
                    self._playback_time=self._playback_next_frame=None
                else:
                    self._stop()
                return False
            # The preceding tick displayed the completed block at exactly 100%.
            self.render_delta_index+=1
            self._sync_progress(0.0)
            self._playback_time=None
            return True
        if self._playback_time is None:
            self._playback_time=now
            self._playback_next_frame=now+1.0/30.0
            return False
        elapsed=max(0.0,now-self._playback_time)
        d=self.sim.deltas[self.render_delta_index]
        duration=float(np.clip(max(d.sweep.move_length,1.0)/45.0,0.35,1.6))
        self._sync_progress(min(1.0,self.progress+elapsed*self._speed_updates.applied/duration))
        self._playback_time=now
        # Keep absolute 30 Hz deadlines. GUI tick jitter must not accumulate;
        # if rendering overruns, skip expired deadlines instead of queueing work.
        interval=1.0/30.0
        deadline=self._playback_next_frame
        missed=max(1,int(np.floor((now-deadline+1e-9)/interval))+1)
        self._playback_next_frame=deadline+missed*interval
        if (self.progress>=1.0 and self.render_delta_index==len(self.sim.deltas)-1
                and not self._playback_has_pending_blocks()):
            self._stop()
        return True

    def _playback_has_pending_blocks(self):
        return False

    def _play(self):
        if not self.sim.deltas or self._playing: return
        # Playback is motion-oriented, so temporarily leave algorithm-stage mode.
        self.cut_debug.checked=False
        u=self._progress_updates.pending
        if u is None: u=self.progress
        self.render_delta_index=max(0,min(self.render_delta_index,len(self.sim.deltas)-1))
        if u>=1.0:
            self.render_delta_index=(self.render_delta_index+1)%len(self.sim.deltas)
            u=0.0
        self._sync_progress(u)
        self._playback_settings=self._render_settings(half_clip=bool(self.half_gs.checked))
        if not self._playback_attributes.ready(self.sim.deltas[self.render_delta_index],self._playback_settings):
            self._playback_prepare_index=self.render_delta_index
            self._playback_time=self._playback_next_frame=None
            self.status.text='Preparing 30 fps playback...'
            self._playing=True
            return
        # Prepare and display the first frame before starting the motion clock.
        self._redraw()
        self._playback_time=time.perf_counter()
        self._playback_next_frame=self._playback_time+1.0/30.0
        self._playing=True

    def _reset(self):
        self._stop(); self._reset_simulation(); self._half_material_cache.clear()
        self._render_attribute_cache.clear()
        self._playback_attributes.clear()
        self.render_delta_index=-1; self.progress=0.0; self.debug_stage='Before'
        self._sync_progress(0.0); self.stage_label.text='Stage: Before'
        self.show_removed.checked=False; self.show_split.checked=False
        self.show_added_direct.checked=False; self.show_added_grid.checked=False; self.show_vertex_refine.checked=False; self.show_final.checked=False
        self._redraw()

    def _reset_simulation(self):
        self.sim.reset()

    def _add_tool_visual(self,d):
        if d is None or not self.show_tool.checked:
            self._tool_scene.set_visible(False)
            return
        u=(0.0 if (self.cut_debug.checked and self.debug_stage=='Before') else (1.0 if self.cut_debug.checked else float(self.progress)))
        start,end=command_endpoints(d.command)
        spec=tool_model_spec(d.command.tool_type,d.command.radius,self.sim.stock_lo,self.sim.stock_hi,d.command.cutting_range)
        self._tool_scene.set_pose(spec,start+(end-start)*u,d.command.tool_axis)

    def _set_tool_visible(self,value):
        self.show_tool.checked=bool(value)
        if hasattr(self,'debug_tool'):
            self.debug_tool.checked=bool(value)
        if hasattr(self,'workspace_tool'):
            self.workspace_tool.checked=bool(value)
        self._add_tool_visual(self._current_delta())
        self._request_redraw()

    def _add_splat_geometry(self,name,cloud,material,settings,solid_color=None,offset_scale=0.0,attributes=None):
        if cloud is None and attributes is None: return None
        if attributes is None and not len(cloud.positions):
            return self._render_attribute_cache.get(cloud,material,settings)
        o3d=self.o3d
        draw_cloud=cloud
        if abs(float(offset_scale))>1e-12 and not np.any(cloud.inherited_clip[:,:,:,:3]) and all(x is None for x in cloud.sweep_domains):
            pos=cloud.positions+cloud.normals*(cloud.support*float(offset_scale))[:,None]
            draw_cloud=SplatCloud(pos,cloud.normals.copy(),cloud.support.copy(),cloud.tool_ids.copy(),
                                  cloud.refine_levels.copy(),cloud.seam_weight.copy(),cloud.inherited_clip.copy(),cloud.sweep_domains.copy())
        # Cheap solid-colour overlays must not evict expensive main-scene fits.
        arr=attributes
        if arr is None:
            arr=(self._render_attribute_cache.get(draw_cloud,material,settings) if solid_color is None
                 else gaussian_attributes.pack_gaussian_attributes(draw_cloud,material,settings))
        if 'clip_pending' in arr:
            draw=playback_draw_attributes(arr) if attributes is not None else native_draw_attributes(arr)
        else:
            draw=split_draw_attributes(arr)
        draw=self._prepare_surface_draw(draw,settings,solid_color)
        if name=='work': self._last_work_draw=draw
        if len(draw['positions']):
            self._configure_gs_capacity(len(draw['positions']))
            pc=gaussian_tensor_cloud(o3d,draw)
            self._gs_scene.add(name,pc,self.gs_mat)
            if name!='work': self._debug_names.append(name)
        return arr

    def _save_render_case(self):
        """Freeze the current upload and camera without calculating any shape."""
        from pathlib import Path
        from datetime import datetime
        from cutsim.diagnostics.render_probe import save_case
        self._stop()
        draw=getattr(self,'_last_work_draw',None)
        if draw is None:
            self.status.text+='\nNo native GS upload available to save.'
            return
        path=Path(__file__).resolve().parents[2]/'validation-results'/'render-cases'/(
            datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.npz')
        save_case(path,draw,self.scene.scene.camera,self.scene.frame.width,self.scene.frame.height,
                  self.gs_mat,context=dict(block=self.render_delta_index,progress=self.progress,
                                           stage=self.debug_stage,cut_debug=self.cut_debug.checked,
                                           gs_visible=self._gs_scene.visible))
        self.status.text+='\nSaved work GS + camera: '+str(path)
        print('Frozen render case: '+str(path),flush=True)
        self.window.post_redraw()

    def _add_two_edge_markers(self,arr):
        centres=_two_edge_centres(arr,exclude_multi=bool(self.show_multi_edge.checked))
        if not len(centres): return
        pc=self.o3d.geometry.PointCloud()
        pc.points=self.o3d.utility.Vector3dVector(centres)
        pc.paint_uniform_color([0.,1.,1.])
        self.scene.scene.add_geometry('debug_two_edge_centres',pc,self.candidate_mat)
        self._debug_names.append('debug_two_edge_centres')

    def _add_multi_edge_markers(self,arr):
        centres=_multi_edge_centres(arr)
        if not len(centres): return
        pc=self.o3d.geometry.PointCloud()
        pc.points=self.o3d.utility.Vector3dVector(centres)
        pc.paint_uniform_color([1.,0.,1.])
        self.scene.scene.add_geometry('debug_multi_edge_centres',pc,self.candidate_mat)
        self._debug_names.append('debug_multi_edge_centres')

    def _redraw(self):
        o3d=self.o3d
        self._remove('work')
        self._last_work_draw=None
        for n in self._debug_names: self._remove(n)
        self._debug_names=[]
        d=self._current_delta()
        # Prepared playback attributes already contain all selected GS records.
        # Avoid copying a second full SplatCloud (including unused clip domains).
        motion=d is not None and not self.cut_debug.checked
        visible=None if motion else self._visible_cloud()

        use_half=bool(self.half_gs.checked)
        material=(self._material_for_progress(d,self.progress) if use_half else self.sim.material)
        settings=self._render_settings(half_clip=use_half)
        attributes=None
        if motion:
            self._playback_attributes.prepare(d,settings,self._render_attribute_cache,
                self._material_snapshot(d.block_index),self._material_snapshot(d.block_index+1))
            attributes=self._playback_attributes.get(self.progress)
            # Count the selected frozen records for status; do not refit boundaries.
            _native_clip_stats(attributes,material,local_updated_points=0,
                local_prepared_points=0,local_reused_points=len(attributes['positions']))
            material.last_edge_phase_ms={}
        visible_count=len(attributes['positions']) if motion else len(visible.positions)
        arr=self._add_splat_geometry('work',visible,material,settings,attributes=attributes)
        edge_status=_edge_split_status(material)
        unresolved=clip_marker_mask(arr)
        size_approximate=size_approximation_mask(arr)
        if np.any(unresolved):
            pc=self.o3d.geometry.PointCloud()
            pc.points=self.o3d.utility.Vector3dVector(arr['positions'][unresolved])
            pc.paint_uniform_color([1.,0.,1.])
            self.scene.scene.add_geometry('inherited_clip_pending',pc,self.point_mat)
            self._debug_names.append('inherited_clip_pending')
            print(f"[PERSISTENT CLIP] {int(unresolved.sum())} GS remain unresolved after bounded render subdivision; marked magenta.")
        if 'clip_pending' in arr:
            approximate_count=int(np.count_nonzero(arr['clip_approximated']))
            if approximate_count and getattr(self,'_last_approx_warning',None)!=(id(arr),approximate_count):
                print(f'[CLIP SELECT] {approximate_count} parents use a reduced set of boundaries (0-2); candidates and component scores retained in arr.approximated. Omitted boundaries may leak.')
                self._last_approx_warning=(id(arr),approximate_count)

        self.show_two_edge.enabled=use_half
        if use_half and self.show_two_edge.checked:
            self._add_two_edge_markers(arr)
        self.show_multi_edge.enabled=use_half
        if use_half and self.show_multi_edge.checked:
            self._add_multi_edge_markers(arr)

        # Algorithm overlays: these are the actual geometric groups, not render-only children.
        if d is not None and self.cut_debug.checked:
            dbg_settings=self._debug_render_settings()
            if self.show_removed.checked:
                self._add_splat_geometry('debug_removed',d.removed,self.sim.material,dbg_settings,[1.00,0.12,0.10],0.06)
            if self.show_split.checked:
                self._add_splat_geometry('debug_split',d.split_children,self.sim.material,dbg_settings,[1.00,0.55,0.08],0.08)
            if self.show_added_direct.checked:
                self._add_splat_geometry('debug_added_direct',d.added_direct,self.sim.material,dbg_settings,[0.10,1.00,0.30],0.06)
            if self.show_added_grid.checked:
                self._add_splat_geometry('debug_added_grid',d.added_grid,self.sim.material,dbg_settings,[0.05,0.78,1.00],0.08)
            if self.show_vertex_refine.checked and len(d.vertex_refined.positions):
                # These are pre-extraction candidates, not reconstructed 2DGS.
                # Preserve their actual coordinates and draw only point markers.
                pc=o3d.geometry.PointCloud()
                pc.points=o3d.utility.Vector3dVector(d.vertex_refined.positions)
                pc.paint_uniform_color([1.00,0.92,0.08])
                self.scene.scene.add_geometry('debug_vertex_refine',pc,self.candidate_mat)
                self._debug_names.append('debug_vertex_refine')
            if self.show_final.checked and self.debug_stage!='Result':
                self._add_splat_geometry('debug_final',self._final_cloud(d),material,self._render_settings(half_clip=use_half),None,0.0)

        if d is not None and self.show_candidates.checked and len(d.candidate_cloud.positions):
            c=d.candidate_cloud; colors=np.tile([0.42,0.42,0.46],(len(c.positions),1))
            colors[d.direct_mask]=[0.18,1.00,0.28]; colors[d.fallback_mask]=[1.00,0.16,0.12]
            pp=c.positions+c.normals*(0.04*c.support[:,None])
            pc=o3d.geometry.PointCloud(); pc.points=o3d.utility.Vector3dVector(pp); pc.colors=o3d.utility.Vector3dVector(colors)
            self.scene.scene.add_geometry('candidate_debug',pc,self.candidate_mat); self._debug_names.append('candidate_debug')

        if d is not None and self.show_grid.checked:
            for i,item in enumerate(d.fallback_items):
                fr,gh,level=item['frame'],float(item['h']),int(item['level'])
                pp,ee=obb_lines(fr,gh); ls=o3d.geometry.LineSet(o3d.utility.Vector3dVector(pp),o3d.utility.Vector2iVector(ee))
                col=DEBUG_LEVEL_COLORS[min(level,len(DEBUG_LEVEL_COLORS)-1)]
                ls.paint_uniform_color(col); name=f'fallback_obb_{i}'; self.scene.scene.add_geometry(name,ls,self.line_mat); self._debug_names.append(name)
                lowers=item.get('debug_crossing_cells_local',np.empty((0,3)))
                if len(lowers):
                    cp,ce,cc=cell_boxes_lines(fr,lowers,gh,col)
                    if len(ce):
                        cl=o3d.geometry.LineSet(o3d.utility.Vector3dVector(cp),o3d.utility.Vector2iVector(ce)); cl.colors=o3d.utility.Vector3dVector(cc)
                        name=f'fallback_cells_{i}'; self.scene.scene.add_geometry(name,cl,self.line_mat); self._debug_names.append(name)
                for kind in ('debug_adaptive_cells','debug_unresolved_cells'):
                    for j,(depth,lowers) in enumerate(item['stats'].get(kind,[])):
                        if not len(lowers):continue
                        color=([1.,.3,.05] if kind=='debug_unresolved_cells' else
                               DEBUG_LEVEL_COLORS[min(level+depth+1,len(DEBUG_LEVEL_COLORS)-1)])
                        cp,ce,cc=cell_boxes_lines(fr,lowers,gh/(2**depth),color)
                        if len(ce):
                            cl=o3d.geometry.LineSet(o3d.utility.Vector3dVector(cp),o3d.utility.Vector2iVector(ce))
                            cl.colors=o3d.utility.Vector3dVector(cc)
                            name=f'{kind}_{i}_{j}';self.scene.scene.add_geometry(name,cl,self.line_mat);self._debug_names.append(name)

        if d is not None and self.show_bounds.checked:
            lo,hi=d.sweep.world_aabb(); pp,ee=world_aabb_lines(lo,hi)
            ls=o3d.geometry.LineSet(o3d.utility.Vector3dVector(pp),o3d.utility.Vector2iVector(ee)); ls.paint_uniform_color([1.0,0.30,0.70])
            self.scene.scene.add_geometry('sweep_aabb',ls,self.line_mat); self._debug_names.append('sweep_aabb')
        if d is not None and getattr(self,'_dexel_layers',None):
            from cutsim.rendering.dexel_debug_overlay import draw_overlay
            draw_overlay(self,d)
        if hasattr(self,'_draw_surface_occupancy'):
            self._draw_surface_occupancy(d)
        self._add_tool_visual(d)

        if d is None:
            self.debug_counts.text='Removed 0 | Split 0 | SweepSurface +0 | LocalGrid +0'
            self.status.text=(f'Computed blocks: 0/{len(self.sim.program)} | Initial splats {visible_count:,}\n'
                              'CUDA Tool-profile Dexel | current GS material\n'
                              f'2DGS display size {self._sigma_updates.applied:.2f}x')
        else:
            s=d.stats
            self.debug_counts.text=(f"Removed {len(d.removed.positions):,} | Split {len(d.split_children.positions):,} | "
                                    f"SweepSurface +{len(d.added_direct.positions):,} | LocalGrid +{len(d.added_grid.positions):,} | 3-face Grid triggers {len(d.vertex_refined.positions):,}")
            self.status.text=(
                f"Block {d.block_index+1}/{len(self.sim.program)} | computed {len(self.sim.deltas)} | "
                f"mode {'ALGORITHM DEBUG' if self.cut_debug.checked else 'MOTION PLAYBACK'} | stage {self.debug_stage}\n"
                f"{_block_computation_status(d)}"
                f"Before {len(d.before.positions):,} -> Removed {len(d.removed.positions):,} -> Split children {len(d.split_children.positions):,} -> Added {len(d.added.positions):,} -> Final {len(self._final_cloud(d).positions):,}\n"
                f"Local Dexel rays {s.get('rays',0):,} | profile rays {s.get('profile_rays',0):,} | changed rays {s.get('changed_rays',0):,}\n"
                f"GS queries: points {s.get('gs_query_points',0):,} | pairs {s.get('gs_query_pairs',0):,} | ray pairs {s.get('gs_ray_pairs',0):,}\n"
                f"Clipping {'ON' if self.half_gs.checked else 'OFF'} | prepared {len(arr['positions']):,} from geometric {visible_count:,} | display {self._sigma_updates.applied:.2f}x\n"
                f"Edge cases: {edge_status}")
        if 'clip_pending' in arr:
            self.status.text+=f'\nNative clip: {len(arr["positions"]):,} parents; {material.last_edge_split_stats.get("clip_draw_count",0):,} draws'
            self.status.text+=f'\nEdge density supplements: {self.sim.last_stats.get("edge_fill",{}).get("added",0) if self.sim.last_stats else 0:,}'
            repair=(self.sim.last_stats or {}).get('sweep_repair',{})
            self.status.text+=f'\nSweepSurface repair: {repair.get("added",0):,} GS; unresolved probes {repair.get("unresolved",0):,}; {repair.get("milliseconds",0.):.2f} ms'
            self.status.text+=f'\nPartial parents kept at original size: {self.sim.last_stats.get("clip_preserved_parents",0) if self.sim.last_stats else 0:,}'
            self.status.text+=f'\nShared ball intersection planes: {self.sim.last_stats.get("curve_boundary_gs",0) if self.sim.last_stats else 0:,}'
            self.status.text+=f'\nSweep Top-2: {int(arr["clip_sweep_top2"].sum()):,}; empty: {int(arr["clip_empty"].sum()):,} [hidden / no marker]'
            self.status.text+=f'\nRender subdivision: {material.last_edge_split_stats.get("clip_fallback_parents",0):,} parents / {material.last_edge_split_stats.get("clip_fallback_children",0):,} child GS; pruned {material.last_edge_split_stats.get("clip_fallback_pruned",0):,}; size stops {material.last_edge_split_stats.get("clip_fallback_size_stopped",0):,}'
            self.status.text+=f'\nPersistent clip unresolved: {int(unresolved.sum()):,} [MAGENTA]'
            self.status.text+=f'\nMin-size Top-2 approximation: {int(size_approximate.sum()):,} parents [no marker]'
            self.status.text+=f'\nConcave splits {int(arr["clip_concave"].sum()):,}; uncertain joins {int(arr["clip_join_uncertain"].sum()):,}'
            self.status.text+=f'; reduced boundary set {np.count_nonzero(arr["clip_approximated"]):,}'
            self.status.text+=f'\nBoundary candidates {int(arr["clip_ridge_count"].sum()):,}; effective {int(arr["clip_eligible_count"].sum()):,}'
            self.status.text+=f'\n2 applied edge centres: {np.count_nonzero(arr["clip_edge_count"]==2):,}'
            self.status.text+=(' [CYAN; 3+ MAGENTA priority]' if self.show_multi_edge.checked else ' [CYAN]') if use_half and self.show_two_edge.checked else ' (markers OFF)'
            self.status.text+=f'\n3+ boundary centres: {np.count_nonzero(arr["clip_candidate_count"]>=3):,}'
            self.status.text+=(' [MAGENTA]' if use_half and self.show_multi_edge.checked else ' (markers OFF)')
            if getattr(self,'_native_upload_error',''): self.status.text+='\n'+self._native_upload_error
        self._request_redraw()

    def _create_window(self,title,width,height):
        from cutsim.ui.native_window import VIEWPORT_BACKGROUND
        window = self.gui.Application.instance.create_window(title,width,height)
        window.renderer.set_clear_color(VIEWPORT_BACKGROUND)
        return window

    def run(self): self.gui.Application.instance.run()

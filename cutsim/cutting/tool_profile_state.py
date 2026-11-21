"""Live GS state and transient ray-grid helpers; no material-history engine."""
from threading import RLock
from time import perf_counter
from types import SimpleNamespace
import numpy as np
from cutsim.geometry.sampling_density import sampling_pitch,validate_gs_density
from cutsim.core.splat_cloud import analytic_box_surface_splats
from cutsim.geometry.frames import tangent_frames,rotation_matrices_to_quaternions
from cutsim.rendering.profile_surface_identity import analyze_profile_continuity
from cutsim.rendering.profile_demo_gs import GaussianSnapshot,draw_attributes
from cutsim.rendering.profile_cut_clip import clipped_draw
from cutsim.cuda.profile_program import ProgramSnapshot,build_atlas
from cutsim.rendering.gs_extent import support_factor
from cutsim.rendering.gaussian_attributes import machining_tool_colors
RAY_STRIDE=1_000_000

def combine_surfaces(old,keep,new):
    def combine(a,b):
        value=np.concatenate((a[keep],b));value.flags.writeable=False;return value
    ep={k:combine(v,new.endpoints[k]) for k,v in old.endpoints.items()}
    attrs={k:combine(v,new.gaussian.attributes[k]) for k,v in old.gaussian.attributes.items()}
    result=ProgramSnapshot(ep,GaussianSnapshot(attrs,combine(old.gaussian.u,new.gaussian.u),
                                              combine(old.gaussian.v,new.gaussian.v),0.),support_sigma=support_factor(old))
    from cutsim.rendering.analytic_surfaces import combine_metadata
    combine_metadata(old,keep,new,result)
    return result

class ProfileSurfaceState:
    transient_dexels=True
    shape_update_mode="profile_dexel"
    endpoint_policy="first_current"
    join_surface_ids=True
    incremental_clipping=True
    def __init__(self,program,stock_lo,stock_hi,*,gs_density=1.,pitch=None,backend='cuda',max_rays=200000,ray_margin=None,initial_splat=.75):
        if backend!='cuda':raise ValueError('CUDA is required')
        self.program=list(program);self.stock_lo=np.asarray(stock_lo,float);self.stock_hi=np.asarray(stock_hi,float)
        self.gs_density=validate_gs_density(gs_density);self.pitch=pitch;self.backend_request=backend;self.max_rays=max_rays
        self.ray_margin=ray_margin;self.initial_splat=float(initial_splat)
        if self.stock_lo.shape!=(3,) or self.stock_hi.shape!=(3,) or not np.all(np.isfinite([self.stock_lo,self.stock_hi])) or np.any(self.stock_lo>=self.stock_hi):raise ValueError('Invalid stock bounds')
        self.model=None;self.reset()

    def reset(self):
        self.close()
        self._grid_lock=RLock();self._display_grid=None;self._display_block=None
        self._active_grid=None;self.model=None
        self.grid_cell=sampling_pitch(self.initial_splat,self.gs_density)
        self.backend=None;self.fallback_reason=None;self.initialization_ms=0.;self.failed=False
        self.deltas=[];self.step_index=0;self.last_stats=None
        self.surface_continuity=analyze_profile_continuity(self.program)
        self.surface_groups=self.surface_continuity.groups if self.join_surface_ids else None
        self._clip_cache=None
        stock=analytic_box_surface_splats(self.stock_lo,self.stock_hi,self.grid_cell)
        p,n=stock.positions,stock.normals;u,v=tangent_frames(n)
        face=2*np.argmax(abs(n),axis=1)+(np.sum(n,axis=1)>0)
        attrs=dict(positions=p.astype(np.float32),rot=rotation_matrices_to_quaternions(np.stack((u,v,n),axis=2)).astype(np.float32),
            scale=(.75*stock.support).astype(np.float32)[:,None],surface_color=np.ones((len(p),3),np.float32),
            surface_id=(face+1).astype(np.int32)[:,None])
        ep=dict(positions=p,normals=n,owners=(-face-1).astype(np.int64),ray_ids=-np.arange(1,len(p)+1,dtype=np.int64),
                distances=np.zeros(len(p)),tool_ids=np.zeros(len(p),np.int32))
        for value in [*attrs.values(),*ep.values(),u,v]:value.flags.writeable=False
        from cutsim.rendering.initial_stock_clip import box_face_clipping
        clipping=box_face_clipping(p,u,v,attrs['scale'][:,0],self.stock_lo,self.stock_hi)
        self.initial=self.current=ProgramSnapshot(ep,GaussianSnapshot(attrs,u,v,0.),clipping=clipping)
        # Validate configuration on only the first local grid, never the program atlas.
        if self.program:self.display_atlas(0)

    def make_atlas(self,block):
        atlas,_=build_atlas([self.program[block]],self.stock_lo,self.stock_hi,self.gs_density,self.pitch,
            self.max_rays,self.ray_margin,include_stock=False,sweep_envelope=getattr(self,'sweep_envelope',False))
        for pose in atlas.poses:pose['block']=block
        box=atlas.local_boxes[0];atlas.local_boxes=[None]*len(self.program);atlas.local_boxes[block]=box
        return atlas

    @property
    def atlas(self):
        if self._active_grid is not None:return self._active_grid
        return self.display_atlas(min(self.step_index,max(0,len(self.program)-1)))

    def initialize(self):
        if self.backend is not None:return
        started=perf_counter()
        from cutsim.cuda.profile_dexel_cuda import runtime
        runtime()
        self.backend='cuda'
        self.initialization_ms=(perf_counter()-started)*1000

    def close(self):
        model=getattr(self,'model',None)
        if model is not None and hasattr(model,'close'):model.close()
        self.model=None;self._active_grid=None;self._display_grid=None;self._display_block=None

    def compute_all(self):
        while self.step_index<len(self.program):yield self.compute_next()

    def draw(self,snapshot,*,selected=None,size=1.,clip_y=None,tool_colors=True,solid_color=None,clipping=True,cutting_clipping=True):
        count=len(snapshot.endpoints['positions'])
        selected=np.ones(count,bool) if selected is None else np.array(selected,copy=True)
        if clip_y is not None:selected &= snapshot.endpoints['positions'][:,1]<=clip_y+1e-9
        if clipping and cutting_clipping and clip_y is not None and snapshot.clipping is not None and snapshot.clipping.stats.get('clip_subdivision'):
            from cutsim.rendering.clip_subdivision import section_snapshot
            snapshot=section_snapshot(snapshot,selected,clip_y)
            selected=np.ones(len(snapshot.endpoints['positions']),bool);clip_y=None
        context=SimpleNamespace(metadata=dict(stock_low=self.stock_lo,stock_high=self.stock_hi,shift=np.zeros(3),rotation=np.eye(3)))
        stock_color = (.62, .62, .62)
        if clipping and cutting_clipping and snapshot.clipping is not None:
            draw=clipped_draw(snapshot,selected,size,clip_y)
            draw['surface_color'][:] = stock_color
        else:draw=draw_attributes(context,snapshot.gaussian,selected,stock_color,size,clip_y)
        tools=snapshot.endpoints['tool_ids'][selected]
        cut = tools > 0
        if solid_color is not None:
            draw['surface_color'][:] = solid_color
        elif tool_colors:
            draw['surface_color'][cut] = machining_tool_colors(tools[cut])
        else:
            draw['surface_color'][cut] = [1., .62, .22]
        if not clipping:
            for slot in (1,2):draw[f'clip{slot}_angle'].fill(4.);draw[f'clip{slot}_split'].fill(0.)
        return draw

"""Tool-profile grid and immutable GS snapshot formats."""
from dataclasses import dataclass,replace,field
from time import perf_counter
from cutsim.rendering.gs_extent import support_factor
from types import SimpleNamespace

import numpy as np

from cutsim.core.commands import command_endpoints,command_sweep
from cutsim.core.splat_cloud import SplatCloud,analytic_box_surface_splats
from cutsim.cuda.tool_profile_grid import ToolProfileGrid
from cutsim.cuda.swept_profile_grid import swept_grids
from cutsim.cuda.profile_ray_domain import intersect_box
from cutsim.cutting.local_dexel import bounds as sweep_bounds
from cutsim.geometry.frames import tangent_frames,rotation_matrices_to_quaternions
from cutsim.geometry.sampling_density import sampling_pitch,validate_gs_density
from cutsim.geometry.grid_sizing import GridSizingSettings
from cutsim.rendering.profile_demo_gs import GaussianSnapshot,profile_sigma_arrays,draw_attributes
from cutsim.rendering.gs_schema import validate_gaussian_attributes
from cutsim.rendering.profile_cut_clip import prepare_cut_clipping,clipped_draw




@dataclass
class ProfileAtlas:
    origins: np.ndarray
    directions: np.ndarray
    charts: np.ndarray
    cell_bounds: np.ndarray
    radii: np.ndarray
    pitches: np.ndarray
    profile_count: int
    poses: list
    ray_bounds: np.ndarray | None = None
    surface_distance: np.ndarray | None = None
    local_boxes: list | None = None
    feeds: np.ndarray | None = None

    def points(self,ray_ids,distance):
        return self.origins[ray_ids]+np.asarray(distance)[...,None]*self.directions[ray_ids]


def build_atlas(program,low,high,density=1.,pitch=None,max_rays=200000,ray_margin=None,*,include_stock=True,sweep_envelope=False):
    if pitch is not None and (not np.isfinite(pitch) or pitch<=0):raise ValueError('Profile pitch must be positive')
    if ray_margin is not None and (not np.isfinite(ray_margin) or ray_margin<=0):raise ValueError('Profile ray margin must be positive')
    if not isinstance(max_rays,int) or not 100<=max_rays<=500000:raise ValueError('Ray budget must be 100..500000')
    density=validate_gs_density(density);grids=[];feeds=[];poses=[];seen=set();total=0;local_boxes=[]
    for block,command in enumerate(program):
        if command.cutting_range is None:raise ValueError('Profile mode requires a finite cutting_range on every tool')
        a,b=command_endpoints(command)
        sweep=command_sweep(command,high[2])
        h=(command.gs_sampling.resolve_sweep_pitch(sweep,GridSizingSettings(),density)
           if pitch is None else sampling_pitch(pitch,density))
        angle=command.gs_sampling.resolve_ball_angle(density)
        margin=h if ray_margin is None else ray_margin
        box_low,box_high=sweep_bounds(sweep)
        # Halo keeps the physical cut boundary strictly inside the window.
        box_low=np.maximum(low,box_low-margin);box_high=np.minimum(high,box_high+margin)
        local_boxes.append((box_low,box_high))
        if np.any(box_low>=box_high):continue
        for grid,feed,part in swept_grids(command,h,margin,max_rays,ball_angle_deg=angle,
                                         sweep_envelope=sweep_envelope,stock_top=high[2],
                                         stock_bounds=(low,high),sweep=sweep):
            tip=grid.tip
            key=(block,part,*np.round(tip,12))
            if key in seen:continue
            seen.add(key)
            domain,keep=intersect_box(grid.origins,grid.directions,grid.ray_bounds,box_low,box_high)
            if not keep.any():continue
            # The atlas needs only retained cells, never the discarded rays.
            grid=replace(grid,origins=grid.origins[keep],directions=grid.directions[keep],
                charts=grid.charts[keep],cell_bounds=grid.cell_bounds[keep],
                surface_distance=grid.surface_distance[keep],ray_bounds=domain[keep])
            total+=len(grid.origins)
            if total>max_rays:raise ValueError(f'Profile atlas exceeds {max_rays:,} rays; increase --profile-pitch or --max-profile-rays')
            grids.append(grid);feeds.append(np.broadcast_to(feed,(len(grid.origins),3)))
            poses.append(dict(block=block,tip=tip.tolist(),basis=grid.basis.tolist(),rays=len(grid.origins),pitch=h,kind=grid.kind,part=part,ray_margin_mm=margin,ball_angle_deg=angle,axial_bounds=list(grid.axial_bounds)))
    # Face sampling is only an auxiliary representation of the initial GS
    # boundary. The profile atlas remains the source of new cutting surfaces.
    stock_spacing=sampling_pitch(.75,density)
    stock=analytic_box_surface_splats(low,high,stock_spacing) if include_stock else SplatCloud.empty()
    positive=np.sum(stock.normals,axis=1)>0
    stock_origins=stock.positions[positive].copy();stock_normals=stock.normals[positive]
    stock_axes=np.argmax(stock_normals,axis=1);stock_length=(high-low)[stock_axes]
    stock_origins[np.arange(len(stock_origins)),stock_axes]=low[stock_axes]
    if total+len(stock_origins)>max_rays:raise ValueError('Profile plus stock-normal rays exceed the configured ray budget')
    def join(name,tail):
        return np.concatenate([getattr(g,name) for g in grids]+[tail])
    atlas=ProfileAtlas(join('origins',stock_origins),join('directions',stock_normals),
        join('charts',np.full(len(stock_origins),'stock','U12')),
        join('cell_bounds',np.zeros((len(stock_origins),4))),
        np.concatenate([np.full(len(g.origins),g.radius) for g in grids]+[np.ones(len(stock_origins))]),
        np.concatenate([np.full(len(g.origins),g.pitch) for g in grids]+[np.full(len(stock_origins),stock_spacing)]),
        total,poses,join('ray_bounds',np.column_stack((np.zeros(len(stock_origins)),stock_length))),
        join('surface_distance',stock_length),local_boxes,np.concatenate(feeds+[np.zeros((len(stock_origins),3))]))
    for value in vars(atlas).values():
        if isinstance(value,np.ndarray):value.flags.writeable=False
    return atlas,stock


def local_ray_segments(atlas,block,*,all_rays=False,limit=400):
    """Display the outer half of profile rays, clipped to the local box.

    Material domains/origins stay unchanged; diagnostic interval traces still
    show the full computation range. Cap rays use half their surface distance.
    """
    if block<0 or block>=len(atlas.local_boxes):return np.empty(0,np.int64),np.empty((0,2,3))
    n=atlas.profile_count
    domain,keep=intersect_box(atlas.origins[:n],atlas.directions[:n],atlas.ray_bounds[:n],*atlas.local_boxes[block])
    if not getattr(atlas,'center_rays',False):domain[:,0]=np.maximum(domain[:,0],.5*atlas.surface_distance[:n])
    keep &= domain[:,0]<domain[:,1]
    # Only this move's grid: other moves may overlap this box but have different feeds.
    owned=np.zeros(n,bool);offset=0
    for pose in atlas.poses:
        if pose['block']==block:owned[offset:offset+pose['rays']]=True
        offset+=pose['rays']
    keep &= owned
    ids=np.flatnonzero(keep)
    if not all_rays:ids=ids[::max(1,int(np.ceil(len(ids)/limit)))]
    points=atlas.origins[ids,None,:]+domain[ids,:,None]*atlas.directions[ids,None,:]
    return ids,points


@dataclass
class ProgramSnapshot:
    endpoints: dict
    gaussian: GaussianSnapshot
    clipping: object = None
    material_clipping: object = None  # Finite GS footprint; display blending may extend beyond chart seams.
    material_surface_ids: object = None  # Continuity labels independent of display smoothing switches.
    support_sigma: float = field(default_factory=support_factor)
    analytic_patch_ids: object = None
    analytic_patches: object = None  # Live generating geometry, never a command replay list.


@dataclass
class ProfileBlock:
    block_index: int
    command: object
    sweep: object
    before: ProgramSnapshot
    after: ProgramSnapshot
    removed_mask: np.ndarray
    added_mask: np.ndarray
    stats: dict
    adaptive: object = None  # Bounded CPU ray metadata for replay/inspection, never material intervals.
    clipping_debug: object = None  # Optional CPU stage snapshots for the debug UI.
    coverage: object = None  # Bounded finalization diagnostics; no live GPU objects.


def _snapshot(endpoints,atlas,program,*,surface_groups=None,support_scale=1.):
    started=perf_counter();p=endpoints['positions'];n=endpoints['normals']
    u,v=tangent_frames(n);q=rotation_matrices_to_quaternions(np.stack((u,v,n),axis=2))
    rays=endpoints['ray_ids'];owners=endpoints['owners'];cut=owners>=0
    sigma=.75*atlas.pitches[rays]
    if cut.any():
        from cutsim.core.tool_pose import tool_basis
        cut_rows = np.flatnonzero(cut)
        axes, owner_axes = np.unique(np.array([c.tool_axis for c in program]), axis=0, return_inverse=True)
        row_axes = owner_axes[owners[cut]]
        for index in np.unique(row_axes):
            rows = cut_rows[row_axes == index]
            r = rays[rows]
            basis = tool_basis(tuple(axes[index]))
            sigma[rows]=profile_sigma_arrays(atlas.charts[r],atlas.cell_bounds[r],endpoints['distances'][rows],
                atlas.directions[r],n[rows],atlas.radii[r],atlas.pitches[r],
                rotation=basis, feeds=atlas.feeds[r]@basis)
        # Additional Dexels may overlap more widely, but retain the generating
        # cutter's curvature bound. Scale BEFORE clipping probes are generated.
        sigma[cut]*=support_scale
        # A ray belonging to a large tool can hit a later much smaller tool.
        # Bound its footprint by the GENERATING cutter as well as the ray cell,
        # otherwise coarse old grids create large tangent discs on small grooves.
        owner_radius=np.array([c.radius for c in program])[owners[cut]]
        sigma[cut]=np.minimum(sigma[cut],owner_radius/8.)
    axis=np.argmax(np.abs(n),axis=1)
    identity=(1+2*axis+(n[np.arange(len(n)),axis]>0)).astype(np.int32)
    if cut.any():
        kind=np.array([c.tool_type.lower() for c in program])[owners[cut]]
        patch=np.zeros(int(cut.sum()),np.int32)
        axes=np.array([c.tool_axis for c in program])[owners[cut]]
        axial=np.sum(n[cut]*axes,axis=1)
        patch[(kind=='flat')&(axial>1.-1e-8)]=1
        patch[axial<-1.+1e-8]=2
        render_owner=owners[cut] if surface_groups is None else surface_groups[owners[cut]]
        identity[cut]=7+3*render_owner+patch
    attributes=dict(positions=p.astype(np.float32),rot=q.astype(np.float32),scale=sigma.astype(np.float32)[:,None],
                    surface_color=np.ones((len(p),3),np.float32),surface_id=identity[:,None])
    validate_gaussian_attributes(attributes)
    for value in list(endpoints.values())+list(attributes.values()):value.flags.writeable=False
    return ProgramSnapshot(endpoints,GaussianSnapshot(attributes,u,v,(perf_counter()-started)*1000))


def _keys(snapshot):
    ep=snapshot.endpoints
    return np.rec.fromarrays([ep['ray_ids'],ep['owners'],ep['distances']],names='ray,owner,t')

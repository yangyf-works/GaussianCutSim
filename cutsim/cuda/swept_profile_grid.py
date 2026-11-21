"""Tool-profile rays on fixed-axis translation envelopes."""
from dataclasses import replace
import math
import numpy as np
from cutsim.cuda.tool_profile_grid import ToolProfileGrid


def _ramp_grid(command, sweep, pitch, margin, max_rays, stock_bounds, ball_angle_deg):
    from cutsim.geometry.linear_surface import linear_surface_candidates

    lower, upper = sweep.world_aabb() if stock_bounds is None else stock_bounds
    samples = linear_surface_candidates(sweep, pitch, lower, upper, ball_angle_deg=ball_angle_deg,
                                        include_barrel_junctions=True)
    # Generate endpoints only where the cut can create a surface in the workpiece.
    inside = np.all((samples.positions >= lower) & (samples.positions <= upper), axis=1)
    points, directions = samples.positions[inside], -samples.normals[inside]
    count = len(points)
    if count > max_rays:
        raise ValueError('Ramp profile ray budget exceeded; increase the profile pitch')
    if not count:
        return None
    # Normal rays use the verified exterior surface point as their endpoint.
    # The stock and current GS still decide whether that endpoint is material.
    low, high = command.cutting_range.calculation_bounds()
    distance = np.full(count, min(command.radius, .5 * (high-low)))
    cells = np.zeros((count, 4))
    cells[:, [1, 3]] = samples.support[inside, None]
    return ToolProfileGrid(points-distance[:, None]*directions, directions, distance,
                          np.full(count, 'ramp', 'U12'), cells, np.array([0, count]),
                          sweep.a_xyz, np.eye(3), command.radius, pitch, sweep.kind, (low, high),
                          np.column_stack((np.zeros(count), distance+margin)))


def swept_grids(command,pitch,margin,max_rays,*,ball_angle_deg=None,sweep_envelope=False,
                stock_top=None, stock_bounds=None, sweep=None):
    from cutsim.core.commands import command_endpoints, command_sweep
    from cutsim.core.tool_pose import DEFAULT_TOOL_AXIS, box_in_frame
    if command.tool_axis != DEFAULT_TOOL_AXIS:
        from cutsim.core.commands import command_in_tool_frame
        local, basis, origin = command_in_tool_frame(command)
        local_bounds = None if stock_bounds is None else box_in_frame(*stock_bounds, basis, origin)
        for grid, feed, part in swept_grids(local, pitch, margin, max_rays,
                ball_angle_deg=ball_angle_deg, sweep_envelope=sweep_envelope,
                stock_top=None, stock_bounds=local_bounds):
            yield replace(grid, origins=origin+grid.origins@basis.T,
                          directions=grid.directions@basis.T, tip=origin+grid.tip@basis.T,
                          basis=basis@grid.basis), feed@basis.T, part
        return
    a,b=command_endpoints(command);delta=b-a;length=np.linalg.norm(delta[:2])
    radius=command.radius;low,high=command.cutting_range.start,command.cutting_range.end
    kind=command.tool_type.lower()
    if sweep_envelope and np.array_equal(a[:2], b[:2]):
        tip = a if a[2] <= b[2] else b
        travel = abs(delta[2])
        options = dict(tip=tip, max_rays=max_rays, ray_margin=margin,
                       ball_angle_deg=ball_angle_deg)
        if travel:
            # Use the same axial bounds as the finite analytic cutting solid.
            low, high = command.cutting_range.calculation_bounds()
        top = high + travel
        # A cap coincident with or above the stock top cannot create a new
        # material ceiling. Do not use its rays as replacement evidence.
        include_top = bool(travel and (stock_top is None or tip[2]+top < stock_top))
        if kind == 'flat' or high >= radius or not travel:
            grid = ToolProfileGrid.build(kind, radius, (low, top), pitch,
                                         include_upper_cap=include_top, **options)
            yield grid, np.zeros(3), 'plunge_envelope'
        else:
            # Below the equator, the top disk of the lower ball pose is the
            # largest section. Translate that disk to form the side and cap.
            ball = ToolProfileGrid.build(kind, radius, (low, high), pitch, **options)
            yield ball, np.zeros(3), 'plunge_ball'
            effective = math.sqrt(high*(2*radius-high))
            extension = ToolProfileGrid.build('flat', effective, (high, top), pitch,
                include_lower_cap=False, include_upper_cap=include_top, **options)
            yield extension, np.zeros(3), 'plunge_extension'
        return
    if sweep_envelope and length > 0. and delta[2] != 0.:
        if sweep is None:
            sweep = command_sweep(command, math.inf if stock_top is None else stock_top)
        grid = _ramp_grid(command, sweep, pitch, margin, max_rays, stock_bounds, ball_angle_deg)
        if grid is not None:
            yield grid, np.zeros(3), 'ramp_envelope'
        return
    if length<1e-12 or abs(delta[2])>1e-12:
        # The diagnostic non-envelope mode samples individual tool poses.
        steps=max(1,int(np.ceil(np.linalg.norm(delta)/max(radius,pitch))))
        for f in np.linspace(0,1,steps+1):
            yield ToolProfileGrid.build(kind,radius,(low,high),pitch,tip=a+f*delta,
                max_rays=max_rays,ray_margin=margin,ball_angle_deg=ball_angle_deg),np.zeros(3),'pose'
        return
    feed=delta/length;side=np.array([-feed[1],feed[0],0.]);z=np.array([0.,0.,1.])
    for tip,sign,label in ((a,-1.,'start'),(b,1.,'end')):
        grid=ToolProfileGrid.build(kind,radius,(low,high),pitch,tip=tip,max_rays=max_rays,ray_margin=margin,ball_angle_deg=ball_angle_deg)
        keep=sign*((grid.surface_points-tip)@feed)>=-1e-12
        yield replace(grid,**{k:getattr(grid,k)[keep] for k in
            ('origins','directions','surface_distance','charts','cell_bounds','ray_bounds')}),feed,label
    # Cross-section cells are extruded along the feed; no full circles inside.
    cross=[]
    def cells(lo,hi):
        if hi<=lo:return []
        count=max(1,math.ceil((hi-lo)/pitch))
        if count>max_rays:raise ValueError('Swept profile ray budget exceeded')
        edges=np.linspace(lo,hi,count+1)
        return list(zip(edges[:-1],edges[1:]))
    if kind=='ball' and low<radius:
        theta0=math.acos(np.clip(1-low/radius,0,1));theta1=math.acos(np.clip(1-min(high,radius)/radius,0,1))
        if ball_angle_deg is None:
            ball_cells=cells(radius*theta0,radius*theta1)
        else:
            from cutsim.geometry.tool_section import ball_section_angles
            edges=radius*ball_section_angles(radius,low,high,pitch,ball_angle_deg)
            if len(edges)-1>max_rays:raise ValueError('Swept profile ray budget exceeded')
            ball_cells=list(zip(edges[:-1],edges[1:]))
        for sign in (-1,1):
            for q0,q1 in ball_cells:
                q0,q1=sorted((sign*q0,sign*q1));theta=(q0+q1)/(2*radius)
                cross.append(('strip_ball',q0,q1,radius*z,np.sin(theta)*side-np.cos(theta)*z,radius))
    for q0,q1 in cells(max(low,radius) if kind=='ball' else low,high):
        for sign in (-1,1):cross.append(('strip_side',q0,q1,(q0+q1)/2*z,sign*side,radius))
    def width(height):return math.sqrt(max(0,height*(2*radius-height))) if kind=='ball' and height<radius else radius
    # The upper cutting-range plane remains a volume limit, not a sampled face.
    for height,sign in ((low,-1),):
        for q0,q1 in cells(-width(height),width(height)):
            q=(q0+q1)/2
            bottom=max(low,radius-math.sqrt(max(0,radius**2-q*q))) if kind=='ball' else low
            middle=(bottom+high)/2
            cross.append(('strip_cap',q0,q1,q*side+middle*z,sign*z,abs(height-middle)))
    along=cells(0,length)
    # Shared boundary rows bridge the cap/strip charts. Cell-centre sampling
    # alone leaves no sample on the common tangent, where conservative cutter
    # clipping trims the cap discs. These rays lie exactly on the real sweep
    # boundary and use the same cross-section as the straight middle.
    step=along[0][1]-along[0][0]
    along=[(-step/2,step/2),*along,(length-step/2,length+step/2)]
    if len(along)*len(cross)>max_rays:raise ValueError('Swept profile ray budget exceeded')
    origins=[];directions=[];distances=[];charts=[];bounds=[]
    for s0,s1 in along:
        for chart,q0,q1,o,d,t in cross:
            origins.append(a+(s0+s1)/2*feed+o);directions.append(d);distances.append(t)
            charts.append(chart);bounds.append((s0,s1,q0,q1))
    distance=np.asarray(distances)
    yield ToolProfileGrid(np.asarray(origins),np.asarray(directions),distance,np.asarray(charts,'U12'),
        np.asarray(bounds),np.array([0,len(origins)]),a,np.eye(3),radius,pitch,kind,(low,high),
        np.column_stack((np.zeros(len(origins)),distance+margin))),feed,'middle'

"""Display recorded computation, never reconstruct geometry from the final GS."""
import numpy as np


def segments(trace, name, limit=20000, material_stage='after'):
    # Full query rays are retained in the trace for auditing, but are not
    # material. Draw separate occupied intervals so cavities stay empty.
    if name in ('rays','changed_rays'):
        values=trace[material_stage]
        if name=='changed_rays':
            ids=trace.get(material_stage+'_ray_ids')
            # Older captures lack the interval-to-ray association. Do not
            # substitute full search rays and paint through removed material.
            values=values[:0] if ids is None else values[trace['changed'][ids]]
    elif name=='patch_boxes' and 'patch_edges' in trace:values=trace['patch_edges']
    elif name in ('patch_boxes','update_cells'):
        from cutsim.ui.helpers import world_aabb_lines
        boxes=trace[name]
        selected=boxes[:max(1,limit//12)]
        values=[]
        for lo,hi in selected:
            p,e=world_aabb_lines(lo,hi);values.extend(p[e])
        return np.asarray(values,float).reshape(-1,2,3),len(boxes)*12
    else:values=trace[name]
    return values[:limit],len(values)


def draw_overlay(viewer, delta):
    enabled=getattr(viewer,'_dexel_layers',{})
    if not any(enabled.values()):return
    trace=delta.stats.get('dexel_trace')
    colors={'rays':(.35,.4,.55),'changed_rays':(1.,.15,.1),
            'before':(1.,.65,.1),'after':(.1,1.,.35),
            'patch_boxes':(.1,.8,1.),'update_cells':(.8,.4,1.),'tool_frame':(1.,1.,.2)}
    viewer._dexel_draw_counts={}
    for name,active in enabled.items():
        if not active:continue
        if name=='tool_frame':
            from cutsim.geometry.local_grid import make_step_frame
            from cutsim.ui.helpers import world_aabb_lines
            frame=make_step_frame(delta.sweep,viewer.sim.stock_lo,viewer.sim.stock_hi,0.)
            lo,hi=frame.lo.copy(),frame.hi.copy()
            a,b=delta.sweep.world_aabb();lo[2],hi[2]=a[2],b[2]
            p,e=world_aabb_lines(lo,hi);p=frame.world_from_local(p);lines=p[e];total=len(lines)
        elif trace is not None:
            before=(getattr(getattr(viewer,'cut_debug',None),'checked',False)
                    and getattr(viewer,'debug_stage',None)=='Before')
            lines,total=segments(trace,name,material_stage='before' if before else 'after')
        else:continue
        viewer._dexel_draw_counts[name]=(len(lines),total)
        if not len(lines):continue
        o3d=viewer.o3d;points=lines.reshape(-1,3);edges=np.arange(len(points),dtype=np.int32).reshape(-1,2)
        geom=o3d.geometry.LineSet(o3d.utility.Vector3dVector(points),o3d.utility.Vector2iVector(edges))
        geom.paint_uniform_color(colors[name]);key='dexel_trace_'+name
        viewer.scene.scene.add_geometry(key,geom,viewer.line_mat);viewer._debug_names.append(key)

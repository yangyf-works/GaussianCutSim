"""Project shared cut records to drawing boundaries without changing CPU regions.

Supporting planes enclose the analytic cutter. The complete inscribed conic
sections remain on the CPU. In particular, an own-source cap uses intersecting
GS tangent planes for rendering, avoiding mismatched cap/side approximations.
"""
import copy
import numpy as np

from cutsim.core.regions import ExactSurfaceDomains
from cutsim.geometry.frames import tangent_frames


def drawing_cloud(cloud):
    rows=[i for i,d in enumerate(cloud.sweep_domains) if hasattr(d,'render_domains')]
    if not rows:return cloud
    out=copy.copy(cloud)
    out.sweep_domains=cloud.sweep_domains.copy()
    for i in rows:out.sweep_domains[i]=cloud.sweep_domains[i].render_domains
    return out


def _cap_residual_domains(base, residual, stock=None, height=None):
    """Intersect old draw clips with the computed cap difference, exactly.

    A residual branch often already contains an entire old draw branch.
    In that case it implies the old union: emitting the Cartesian product
    repeats the same constraints and can multiply branches on later passes.
    Compare exact world-plane tuples; preserve the general intersection when
    no branch implies the old union.
    """
    if base is None or not residual:
        return residual
    if stock is not None and len(base)*len(residual)>16:
        # Horizontal stock-face contact. Both operands are unions of convex
        # branches. Join only branches whose polygons can meet in the stock.
        # All residuals contain the stock constraints from effective_regions.
        # Use a relaxed box/planes for rejection only; emit unmodified planes.
        from cutsim.geometry.strip_intersections import intersect
        lo,hi=stock
        scale=max(1.,np.max(np.abs(lo)),np.max(np.abs(hi)))
        collar=64.*np.finfo(np.float32).eps*scale
        x0,y0=lo[:2]-collar
        x1,y1=hi[:2]+collar
        box=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
        def bounds(branch):
            norm=np.linalg.norm(branch[:,:3],axis=1)
            active=np.linalg.norm(branch[:,:2],axis=1)>1e-5*norm
            q=branch[active]
            planes=np.column_stack((q[:,:2],q[:,2]*height+q[:,3]+collar*norm[active]))
            poly=intersect(box,planes)
            # A point/line may not have been clipped by all planes. Keeping
            # its bounds is conservative and preserves zero-thickness contact.
            if not len(poly):return [np.inf,np.inf,-np.inf,-np.inf]
            return np.r_[poly.min(axis=0)-collar,poly.max(axis=0)+collar]
        old_bounds=np.asarray([bounds(branch) for branch in base]).reshape(-1,4)
    else:
        old_bounds=None
    base_terms = [set(map(tuple, branch)) for branch in base]
    branches = []
    for branch in residual:
        terms = set(map(tuple, branch))
        if any(old <= terms for old in base_terms):
            branches.append(branch)
        else:
            candidates=base
            if old_bounds is not None:
                x0,y0,x1,y1=bounds(branch)
                hits=(old_bounds[:,0]<=x1)&(old_bounds[:,1]<=y1)&(old_bounds[:,2]>=x0)&(old_bounds[:,3]>=y0)
                candidates=(base[i] for i in np.flatnonzero(hits))
            branches.extend(np.vstack((old, branch)) for old in candidates)
    return ExactSurfaceDomains(branches)


def project_records(cloud, material, records, sources, previous=None, surface_source=None):
    """Cache source-aware drawing boundaries per changed row.

    ``records`` maps output rows to ordered IntersectionRecords. Unchanged
    parents keep their immutable metadata. No material history is intersected
    here and no full-cloud drawing array is constructed. Smooth source curves
    are refined using the source pairs already identified by those records.
    """
    from cutsim.rendering.clipping import _assign_sweep_domains, _sweep_support, _curve_nearest_intersections
    from cutsim.core.sweep import BallSweep,FlatSweep
    out=copy.copy(cloud);out.sweep_domains=np.empty(len(cloud.positions),object)
    out.sweep_domains[:]=None
    u,v=tangent_frames(cloud.normals)
    radius=3.*cloud.support*(1.+.2*np.clip(cloud.seam_weight,0.,1.))
    extent=radius[:,None]*np.sqrt(u*u+v*v)
    from cutsim.rendering.stock_clipping import stock_planes,merge_stock,merge_boundaries
    for i,p in enumerate(cloud.positions):
        if previous is not None:
            old=previous[i]
            out.sweep_domains[i]=getattr(old,'render_domains',old)
        distance=np.minimum(p-material.stock_lo,material.stock_hi-p)
        if np.any(((extent[i]>1e-8)&(distance<extent[i]+1e-7))|(distance< -1e-7)):
            out.sweep_domains[i]=merge_stock(out.sweep_domains[i],stock_planes(material,p,cloud.normals[i]))
    groups={}
    for row,items in records.items():
        for record in items:
            if record.state!='cut' and not record.cutter_sections:continue
            groups.setdefault(record.source_index,[]).append((row,record))
    for sid,items in sorted(groups.items()):
        source=sources[sid];hints={}
        ordinary=[]
        for row,record in items:
            if getattr(record,'material_cap_contact',False):
                # At a coincident finite cutting cap the general supporting-
                # plane renderer calls the whole GS plane "outside". Reuse
                # the actual oriented planar difference already calculated by
                # cutting, including every residual/stock/previous constraint.
                base=out.sweep_domains[row]
                residual=record.domains
                horizontal_stock=(source.kind=='ball' and
                    getattr(source,'cutting_end',None)==source.radius and
                    record.origin[2]==material.stock_hi[2] and
                    np.array_equal(record.normal,(0.,0.,1.)))
                stock=(material.stock_lo,material.stock_hi) if horizontal_stock else None
                out.sweep_domains[row]=_cap_residual_domains(base,residual,stock,record.origin[2])
            else:ordinary.append((row,record))
        items=ordinary
        if not items:continue
        for row,record in items:
            # Plane-section vertices encode the polygonal CPU approximation.
            # Using their normals as the drawing candidate set exposes those
            # facets at smooth boundaries. Reuse the record's cutter identity
            # and the existing draw-only support scorer; this evaluates local
            # normals, not another intersection or a material classification.
            hints[row]=None
        # A finite GS plane and its curved generating surface have different
        # intersections. Refine the latter once using its ALREADY KNOWN source
        # ID; do not rescan history or mistake tangent-plane arc facets for the
        # shared smooth-surface boundary. Keep the refinement in the cut record.
        if isinstance(source,BallSweep):
            owners={};provenance=getattr(material,'_gs_surface_provenance',{})
            for row,record in items:
                owner=surface_source
                if owner is None and hasattr(cloud,'_local_ids'):
                    identity=provenance.get(int(cloud._local_ids[row]),0)
                    if identity>=7:owner=(identity-7)//3
                if owner is not None and owner!=sid and isinstance(sources[owner],BallSweep):
                    owners.setdefault(owner,[]).append((row,record))
            for owner,group in owners.items():
                rows=np.array([r for r,_ in group],np.int64)
                sigma=.75*cloud.support[rows]*(1.+.2*np.clip(cloud.seam_weight[rows],0.,1.))
                planes,valid,points=_curve_nearest_intersections(sources[owner],source,
                    cloud.positions[rows],cloud.normals[rows],sigma)
                for j in np.flatnonzero(valid):
                    row,record=group[j]
                    record.surface_source_index=int(owner)
                    record.surface_curve_point=points[j].copy()
                    record.surface_curve_plane=planes[j].copy()
                    hints[row]=planes[j:j+1]
        _assign_sweep_domains(out,source,np.array([r for r,_ in items],np.int64),
                              material=material,boundary_planes=hints)
    for i,domain in enumerate(cloud.sweep_domains):
        if not isinstance(domain,ExactSurfaceDomains):
            raise TypeError('Drawing projection requires complete CPU regions')
        projected=out.sweep_domains[i]
        # The CPU clips a new lateral face to its generating cutter's axial
        # extent. Replaying only OTHER cuts loses that extent: their outside
        # halfspaces can then grow a wall below its own floor. These exact
        # source limits must survive projection and terminal Top-2 selection.
        owner=surface_source
        if owner is None and hasattr(cloud,'_local_ids'):
            identity=getattr(material,'_gs_surface_provenance',{}).get(int(cloud._local_ids[i]),0)
            if identity>=7:owner=(identity-7)//3
        source=sources[owner] if owner is not None and 0<=owner<len(sources) else None
        source_planes=None
        if isinstance(source,FlatSweep) and abs(cloud.normals[i,2])<1e-12:
            source_planes=np.array([[0.,0.,1.,-source.z_lo],[0.,0.,-1.,source.z_hi]])
            projected=merge_boundaries(projected,source_planes)
        # Preserve every projected constraint until bounded render subdivision.
        # An ordinary tuple would take the early Top-2 path and can expose a
        # third boundary as a visible fin, even inside the finite GS support.
        domain.render_domains=(None if projected is None else ExactSurfaceDomains(projected))
        if source_planes is not None:domain.render_domains.source_planes=source_planes

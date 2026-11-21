"""GaussianCutSim rendering / clipping implementation."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
import numpy as np
import time
from functools import lru_cache
from cutsim.core.local_state import _LocalPointStore, _local_take
from cutsim.core.sweep import BallSweep, FlatSweep, capsule2d_parameter
from cutsim.geometry.frames import rotation_matrices_to_quaternions, tangent_frames
from cutsim.rendering.gaussian_attributes import _edge_stats_empty, _pack_gaussian_attributes_legacy, _render_axes
from cutsim.rendering.surface_identity import surface_ids
from cutsim.rendering.gs_schema import draw_from_calculation
from cutsim.rendering.boundary_work import BoundaryRows, PreparedBoundary, BOUNDARY_SETTINGS
from cutsim.rendering.clip_fallback import abandoned_clip_mask, append_pending_fallbacks, build_pending_fallback, pending_fallbacks, size_approximation_mask, unresolved_clip_mask


def validate_clip_planes(planes,count):
    """Validate before upload; never download a device tensor implicitly."""
    if not isinstance(planes,np.ndarray) and hasattr(planes,'device'):
        if str(planes.device)!='CPU:0':
            raise ValueError('clip_planes must be on CPU; transfer explicitly first')
        planes=planes.numpy()
    data=np.asarray(planes)
    if data.ndim!=3 or data.shape[0]!=count or data.shape[1] not in (1,2) or data.shape[2]!=3:
        raise ValueError('clip_planes must have shape (N,1|2,3); more than two edges cannot be truncated')
    with np.errstate(over='ignore',invalid='ignore'):
        data=np.ascontiguousarray(data,dtype=np.float32)
    if not np.all(np.isfinite(data)):
        raise ValueError('clip_planes must remain finite after Float32 conversion')
    return data


def world_clip_to_local(directions,clearance,rot,scale,*,axes=None):
    """d.(x-centre)+clearance>=0 -> a*u+b*v+c>=0, normalized by linear scale."""
    directions=np.asarray(directions,dtype=np.float64).reshape(-1,3)
    clearance=np.asarray(clearance,dtype=np.float64).reshape(-1)
    if len(directions)!=len(clearance): raise ValueError('boundary count mismatch')
    u,v=(_render_axes(np.asarray(rot,dtype=float).reshape(1,4)) if axes is None else axes)
    return np.column_stack((directions@u[0]*scale[0],directions@v[0]*scale[1],clearance))


CLIP_ATTRIBUTE_NAMES=('clip1_angle','clip1_split','clip2_angle','clip2_split')


def local_planes_to_clip_values(planes):
    """Encode generated local constraints without altering the Gaussian frame.

    This encodes application geometry as native drawing clip values.
    More than two input constraints are rejected before examining their values.
    """
    planes=np.asarray(planes,dtype=np.float64)
    if planes.ndim!=2 or planes.shape[1]!=3 or len(planes)>2:
        raise ValueError('Expected at most two local (a,b,c) constraints; never truncate edges')
    if not np.isfinite(planes).all(): raise ValueError('Boundary coefficients must be finite')
    values=np.array([4.,0.,4.,0.],dtype=np.float32); active=0
    for a,b,c in planes:
        length=np.hypot(a,b)
        if length==0:
            if c<0:
                # Contradictory constraints over the finite Gaussian footprint.
                return np.array([0.,1.,-np.pi,1.],dtype=np.float32)
            continue
        angle=np.float32(np.arctan2(b,a))
        if angle>=np.float32(np.pi): angle=-np.float32(np.pi)
        with np.errstate(over='ignore',invalid='ignore'):
            split=np.float32(-c/length)
        if not np.isfinite(split): raise ValueError('Boundary split must fit finite Float32')
        values[2*active:2*active+2]=(angle,split); active+=1
    return values


class NativeClipArrays(dict):
    """Full parent attributes and all reduced-selection diagnostics remain here."""
    pending=None
    approximated=None  # row -> all candidates, selection, weights; never uploaded


def native_draw_attributes(arr):
    """Expand concave regions and cached pending-domain children for rendering."""
    _resolve_pending_clip_draws(arr)
    names=('positions','scale','rot','coverage_logit','surface_color','surface_id')+CLIP_ATTRIBUTE_NAMES
    joins=np.asarray(arr.get('clip_concave',np.zeros(len(arr['positions']),bool)),dtype=bool)
    pending=np.asarray(arr.get('clip_pending',np.zeros(len(joins),bool))) | arr.get('clip_empty',False)
    if not np.any(joins):
        return append_pending_fallbacks(draw_from_calculation({k:arr[k][~pending] for k in names if k in arr}),arr)
    counts=1+joins.astype(np.int64)
    parent_ids=np.repeat(np.arange(len(joins),dtype=np.int64),counts)
    out={k:arr[k][parent_ids].copy() for k in names if k in arr}
    second_rows=np.cumsum(counts)[joins]-1
    values=arr['clip_second_values'][joins]
    for column,name in enumerate(CLIP_ATTRIBUTE_NAMES):out[name][second_rows,0]=values[:,column]
    visible=~pending[parent_ids]
    return append_pending_fallbacks(draw_from_calculation({k:value[visible] for k,value in out.items()}),arr)


def _flat_sweep_face_domains(sw,p,n,h,cut_surface):
    """Identify a generating flat face using position AND oriented face normal.

    A tool ID only establishes cut-surface provenance; it does not identify a
    unique sweep, since one cutter may execute many separate passes.
    """
    _,nearest=capsule2d_parameter(p[:,:2],sw.a_xy,sw.b_xy)
    rv=p[:,:2]-nearest; radius=np.linalg.norm(rv,axis=1)
    radial=radius-sw.radius
    gr=np.zeros_like(p); gr[:,:2]=rv/np.maximum(radius[:,None],1e-30)
    # At the capsule axis, a stable radial supporting line is sufficient; the
    # line is normally outside this Gaussian's footprint and is discarded.
    move=sw.b_xy-sw.a_xy; length=float(np.linalg.norm(move))
    fallback=np.array([1.,0.]) if length<1e-15 else np.array([-move[1],move[0]])/length
    gr[radius<1e-14,:2]=fallback
    tol=np.maximum(.04*h,1e-6); alignment=math.cos(math.radians(25.))
    bottom=cut_surface&(np.abs(p[:,2]-sw.z_lo)<=tol)&(radial<=tol)&(n[:,2]>=alignment)
    top=cut_surface&(np.abs(p[:,2]-sw.z_hi)<=tol)&(radial<=tol)&(-n[:,2]>=alignment)
    side=(cut_surface&(np.abs(radial)<=tol)&(p[:,2]>=sw.z_lo-tol)&
          (p[:,2]<=sw.z_hi+tol)&(np.sum(-gr*n,axis=1)>=alignment))
    z=np.broadcast_to(np.array([0.,0.,1.]),p.shape)
    # Each value <= 0 defines the domain of the identified face. These are
    # face-domain constraints, not the outside-of-cutter constraint used for
    # a DIFFERENT cutting sweep.
    domains=[(radial,gr,bottom|top,True),
             (sw.z_lo-p[:,2],-z,side,True),
             (p[:,2]-sw.z_hi,z,side,True)]
    return bottom|top|side,domains


def _gs_radius(support,seam_weight,settings):
    """Finite intersection disk; each Gaussian uses its own rendered sigma."""
    value=float(settings.intersection_sigma)
    if not np.isfinite(value) or not .5<=value<=3.:
        raise ValueError('Intersection radius must be between 0.5 and 3 sigma')
    factor=1.+np.clip(np.asarray(seam_weight),0.,1.)*(max(float(settings.resolution_seam_scale),1.)-1.)
    sigma=(np.asarray(support)*float(settings.sigma_scale)*factor).astype(np.float32)
    # Preserve the tested arithmetic order, including the 3-sigma baseline.
    radius=3.*sigma.astype(np.float64)
    return radius if value==3. else radius*(value/3.)


def _geometric_ridge_pairs(p,n,r,q,m,s,min_angle_deg=5.,center_band_fraction=0.,valley_min_angle_deg=5.,clip_valleys=True):
    """One parent vs neighbours: independent positions/normals, finite disks.

    Normals are oriented material-outward. Negative signed distances BOTH ways
    define a strict ridge. The test optionally tolerates a small one-sided
    centre residual from sampled surfaces; this is an inference, not a
    strict ridge proof. Valleys use the sign-mirrored test; coincident pairs are excluded.
    Plane intersection must have a positive-length segment inside BOTH disks.
    No tool, material SDF, history or pre-existing clip direction is involved.
    """
    p=np.asarray(p,dtype=float); n=np.asarray(n,dtype=float)
    n=n/max(float(np.linalg.norm(n)),1e-30)
    q=np.asarray(q,dtype=float).reshape(-1,3); m=np.asarray(m,dtype=float).reshape(-1,3)
    m=m/np.maximum(np.linalg.norm(m,axis=1)[:,None],1e-30)
    s=np.asarray(s,dtype=float).reshape(-1)
    delta=q-p; cosine=np.clip(m@n,-1.,1.)
    sine2=np.maximum(0.,1.-cosine*cosine)
    da=delta@n; db=-np.sum(m*delta,axis=1)
    tolerance=1e-7*np.maximum(np.maximum(r,s),1e-5)
    valid=_edge_signed_mask(da,db,r,s,center_band_fraction,cosine=cosine,valley_min_angle_deg=valley_min_angle_deg,clip_valleys=clip_valleys)&(sine2>=math.sin(math.radians(min_angle_deg))**2)
    valid&=np.sum(delta*delta,axis=1)<=(r+s)**2+1e-12
    ids=np.flatnonzero(valid)
    if not len(ids): return ids,np.empty((0,3)),np.empty(0),np.empty(0),np.empty((0,2,3))
    sin=np.sqrt(sine2[ids]); cos=cosine[ids]
    # The kept side follows from the paired centres AFTER ridge classification.
    inward=-(m[ids]-cos[:,None]*n)/sin[:,None]
    clearance=-db[ids]/sin; other_clearance=-da[ids]/sin
    # Valleys retain the opposite half-plane, towards the original GS centre.
    if clip_valleys:
        valley=(~_ridge_signed_mask(da[ids],db[ids],r,s[ids],center_band_fraction))
        inward[valley]*=-1.
        clearance[valley]*=-1.
    # other_clearance is used only squared/absolute in the finite-disk test.
    line=np.cross(n,m[ids])/sin[:,None]
    foot=p-clearance[:,None]*inward
    along=np.sum(delta[ids]*line,axis=1)
    half_a=np.sqrt(np.maximum(0.,r*r-clearance*clearance))
    half_b=np.sqrt(np.maximum(0.,s[ids]**2-other_clearance**2))
    low=np.maximum(-half_a,along-half_b); high=np.minimum(half_a,along+half_b)
    hit=(np.abs(clearance)<r)&(np.abs(other_clearance)<s[ids])&(high-low>tolerance[ids])
    endpoints=np.stack((foot+low[:,None]*line,foot+high[:,None]*line),axis=1)
    opening=np.pi-np.arccos(cos)
    return ids[hit],inward[hit],clearance[hit],opening[hit],endpoints[hit]


def _geometric_ridge_batch(positions,normals,radii,q,m,s,neighbours,settings):
    """Same finite-disk arithmetic, batching elementwise work across parents.

    Keep each parent's original normalization and matrix-vector products: their
    floating-point accumulation order matters near the clipping thresholds.
    """
    positions,normals,radii,q,m,s=(np.asarray(a,dtype=float) for a in (positions,normals,radii,q,m,s))
    chunks=[]
    for row,near in enumerate(neighbours):
        if not len(near): continue
        n=normals[row]/max(float(np.linalg.norm(normals[row])),1e-30)
        peer=m[near]/np.maximum(np.linalg.norm(m[near],axis=1)[:,None],1e-30)
        delta=q[near]-positions[row]
        chunks.append((np.full(len(near),row,np.int64),np.arange(len(near)),
                       np.broadcast_to(positions[row],delta.shape),np.broadcast_to(n,delta.shape),
                       peer,s[near],np.full(len(near),radii[row]),delta,
                       np.clip(peer@n,-1.,1.),delta@n))
    if not chunks:
        return np.empty(0,np.int64),np.empty(0,np.int64),np.empty((0,3)),np.empty(0),np.empty(0),np.empty((0,2,3))
    owner,index,p,n,m,s,r,delta,cosine,da=(np.concatenate(parts) for parts in zip(*chunks))
    sine2=np.maximum(0.,1.-cosine*cosine)
    db=-np.sum(m*delta,axis=1)
    tolerance=1e-7*np.maximum(np.maximum(r,s),1e-5)
    band=settings.edge_center_band
    valid=_edge_signed_mask(da,db,r,s,band,cosine=cosine,valley_min_angle_deg=settings.valley_min_angle_deg,clip_valleys=settings.clip_valleys)
    valid&=sine2>=math.sin(math.radians(settings.half_min_angle_deg))**2
    valid&=np.sum(delta*delta,axis=1)<=(r+s)**2+1e-12
    ids=np.flatnonzero(valid)
    sin=np.sqrt(sine2[ids]); cos=cosine[ids]
    inward=-(m[ids]-cos[:,None]*n[ids])/sin[:,None]
    clearance=-db[ids]/sin; other_clearance=-da[ids]/sin
    if settings.clip_valleys:
        valley=~_ridge_signed_mask(da[ids],db[ids],r[ids],s[ids],band)
        inward[valley]*=-1.;clearance[valley]*=-1.
    line=np.cross(n[ids],m[ids])/sin[:,None]
    foot=p[ids]-clearance[:,None]*inward
    along=np.sum(delta[ids]*line,axis=1)
    half_a=np.sqrt(np.maximum(0.,r[ids]*r[ids]-clearance*clearance))
    half_b=np.sqrt(np.maximum(0.,s[ids]**2-other_clearance**2))
    low=np.maximum(-half_a,along-half_b);high=np.minimum(half_a,along+half_b)
    hit=(np.abs(clearance)<r[ids])&(np.abs(other_clearance)<s[ids])&(high-low>tolerance[ids])
    endpoints=np.stack((foot+low[:,None]*line,foot+high[:,None]*line),axis=1)
    return owner[ids[hit]],index[ids[hit]],inward[hit],clearance[hit],(np.pi-np.arccos(cos))[hit],endpoints[hit]


def _native_boundaries(cloud,material,base,settings,reference_store=None,*,active_rows=None,prune_parallel=True,batch_pairs=True,share_work=False,sparse_signed=True,surface_domains=None):
    """Geometric ridge/valley edges in both surviving GS regions; material is unused.

    A dirty parent reads adjacent reference-only GS from the persistent full
    store. Standalone full packs build the same centre index once. No nearest-K
    truncation: every potentially intersecting local pair is considered.
    The optimization switches retain an unpruned, per-parent reference path
    for exact regression comparisons; production calls use both optimizations.
    Only the packer opts into compact selection results via share_work.
    """
    count=len(cloud.positions); found=[[] for _ in range(count)]
    if not count: return found,np.zeros(0,dtype=np.int32)
    from cutsim.rendering.boundary_domains import BoundaryDomains
    domain_work=BoundaryDomains()
    domains=cloud.sweep_domains if surface_domains is None else surface_domains
    # Explicit sweep domains are authoritative in _pack_native_clip. Their
    # geometric candidate lists would be discarded, but these GS must remain
    # in the reference store as clipping peers for the other parents.
    targets=range(count) if active_rows is None else active_rows
    if not len(targets): return found,np.zeros(count,dtype=np.int32)
    store=reference_store
    if store is None:
        cell=getattr(getattr(getattr(cloud,'_local_state',None),'root',None),'cell_size',None)
        if cell is None: cell=max(4.*float(np.median(cloud.support)),1e-5)
        store=_LocalPointStore(cell)
        indexed=_local_take(cloud,np.arange(count,dtype=np.int64))
        indexed.sweep_domains=np.asarray(domains,dtype=object).copy()
        if not hasattr(indexed,'_local_ids'): indexed._local_ids=np.arange(count,dtype=np.int64)
        store.add(indexed)
    radii=_gs_radius(cloud.support,cloud.seam_weight,settings)
    import os
    if os.environ.get('CUTSIM_INITIAL_PREP_ACCEL','1')!='0':
        from cutsim.rendering.boundary_broadphase import possible_peer_rows
        targets=possible_peer_rows(cloud,targets,store,radii,settings)
    factor=3.000001*float(settings.sigma_scale)*max(1.,float(settings.resolution_seam_scale))
    maximum=factor*store.max_h+1e-6
    groups={}; tested=np.zeros(count,dtype=np.int32)
    own_u_all,own_v_all=_render_axes(base['rot'])
    work=getattr(material,'_boundary_work',None) if share_work else None
    configuration=tuple(getattr(settings,name) for name in BOUNDARY_SETTINGS) if work is not None else None
    for row in targets: groups.setdefault(store.key(cloud.positions[row]),[]).append(row)
    for group in groups.values():
        for begin in range(0,len(group),128):
            batch=np.asarray(group[begin:begin+128],dtype=np.int64)
            p=cloud.positions[batch]; radius=float(radii[batch].max())+maximum
            refs=store.box(p.min(axis=0)-radius,p.max(axis=0)+radius)
            if not len(refs): continue
            records=store.geometry[store.slots(refs)]; q=records['positions']; m=records['normals']
            peer_radii=_gs_radius(records['support'],records['seam_weight'],settings)
            # Apply the existing angle predicate first. A peer parallel to ALL
            # parents in this batch cannot contribute; avoid allocating and
            # evaluating its distance/signed-distance arrays in the first place.
            own_n=cloud.normals[batch]
            own_n=own_n/np.maximum(np.linalg.norm(own_n,axis=1)[:,None],1e-30)
            peer_n=m/np.maximum(np.linalg.norm(m,axis=1)[:,None],1e-30)
            cosine=own_n@peer_n.T
            angled=(1.-cosine*cosine)>=math.sin(math.radians(settings.half_min_angle_deg))**2-1e-12
            columns=(np.flatnonzero(np.any(angled,axis=0)) if prune_parallel else np.arange(len(q)))
            if not len(columns): continue
            records=records[columns]; q=q[columns]; m=m[columns]
            peer_radii=peer_radii[columns]; peer_n=peer_n[columns]
            cosine=cosine[:,columns]; angled=angled[:,columns]
            delta=q[None,:,:]-p[:,None,:]
            distance2=np.sum(delta*delta,axis=2)
            close=distance2<=(radii[batch,None]+peer_radii[None,:])**2+1e-12
            if sparse_signed:
                # Only distance/angle-admitted pairs need the ridge/valley sign
                # test. Keep each dot product's three-component sum unchanged.
                close&=angled
                owner,peer=np.nonzero(close)
                pair_delta=delta[owner,peer]
                da=np.sum(pair_delta*own_n[owner],axis=1)
                db=-np.sum(pair_delta*peer_n[peer],axis=1)
                close[owner,peer]=_edge_signed_mask(da,db,radii[batch[owner]],peer_radii[peer],
                    getattr(settings,'edge_center_band',0.),slack=1e-12,
                    cosine=cosine[owner,peer],valley_min_angle_deg=settings.valley_min_angle_deg,clip_valleys=settings.clip_valleys)
            else:
                # Exhaustive path for exact regression comparisons.
                da=np.sum(delta*own_n[:,None,:],axis=2)
                db=-np.sum(delta*peer_n[None,:,:],axis=2)
                close&=_edge_signed_mask(da,db,radii[batch,None],peer_radii[None,:],getattr(settings,'edge_center_band',0.),slack=1e-12,
                    cosine=cosine,valley_min_angle_deg=settings.valley_min_angle_deg,clip_valleys=settings.clip_valleys)
                close&=angled
            paired=np.flatnonzero(np.any(close,axis=0))
            if not len(paired): continue
            peer_axes=np.empty((len(q),2,3)); peer_scales=np.empty((len(q),2))
            peer_axes[paired],peer_scales[paired]=_edge_peer_frames(m[paired],records['support'][paired],records['seam_weight'][paired],settings)
            peer_positions=q.astype(np.float32).astype(float)
            peer_opacity=float(np.float32(math.log(settings.alpha/(1.-settings.alpha))))
            neighbours=[np.flatnonzero(close[local]) for local in range(len(batch))]
            cached={}; keys={}
            if work is not None:
                for local,row in enumerate(batch):
                    near=neighbours[local]
                    if not len(near): continue
                    # Include source doubles AND rendered frame/rounding. The
                    # ordered peers are exactly those admitted by the unchanged
                    # broad phase, so removals and geometry changes invalidate.
                    own=np.concatenate((cloud.positions[row],cloud.normals[row],
                        [cloud.support[row],radii[row]],base['positions'][row],
                        base['rot'][row],base['scale'][row],base['coverage_logit'][row]))
                    peers=np.column_stack((q[near],m[near],records['support'][near],records['seam_weight'][near]))
                    key=(configuration,own.tobytes(),peers.tobytes(),domain_work.key(domains[row]),
                         tuple(domain_work.key(domain) for domain in records['sweep_domains'][near]))
                    keys[local]=key
                    hit=work.get(key)
                    if hit is not None: cached[local]=hit
            pending=[np.empty(0,np.int64) if local in cached else near for local,near in enumerate(neighbours)]
            if batch_pairs:
                owners,*pair_arrays=_geometric_ridge_batch(p,cloud.normals[batch],radii[batch],q,m,peer_radii,pending,settings)
                boundaries=np.searchsorted(owners,np.arange(len(batch)+1))
            for local,row in enumerate(batch):
                near=neighbours[local]; tested[row]=len(near)
                if not len(near): continue
                if local in cached:
                    found[row]=cached[local]
                    continue
                if batch_pairs:
                    part=slice(boundaries[local],boundaries[local+1])
                    take,directions,clearance,opening,endpoints=(a[part] for a in pair_arrays)
                else:
                    take,directions,clearance,opening,endpoints=_geometric_ridge_pairs(
                        cloud.positions[row],cloud.normals[row],radii[row],q[near],m[near],peer_radii[near],
                        settings.half_min_angle_deg,center_band_fraction=getattr(settings,'edge_center_band',0.),
                        valley_min_angle_deg=settings.valley_min_angle_deg,clip_valleys=settings.clip_valleys)
                if not len(take): continue
                kept,endpoints=domain_work.trim(endpoints,domains[row],records['sweep_domains'][near[take]])
                if not len(kept):continue
                take,directions,clearance,opening=(values[kept] for values in (take,directions,clearance,opening))
                peers=near[take]
                peaks=_edge_density_peak(endpoints,base['positions'][row],
                    np.stack([own_u_all[row],own_v_all[row]]),base['scale'][row,:2],base['coverage_logit'][row,0],
                    peer_positions[peers],peer_axes[peers],peer_scales[peers],peer_opacity,
                    radius_sigma=getattr(settings,'intersection_sigma',3.))
                line=endpoints[:,1]-endpoints[:,0]
                line/=np.maximum(np.linalg.norm(line,axis=1)[:,None],1e-30)
                anchors=endpoints[:,0]+np.sum((q[near[take]]-endpoints[:,0])*line,axis=1)[:,None]*line
                # Merge the same geometric line; union its finite peer segments.
                # Canonical geometry order makes rows independent of stable IDs,
                # cloud ordering, tool IDs and index bucket traversal order.
                order=np.lexsort((opening,clearance,directions[:,2],directions[:,1],directions[:,0]))
                rows=BoundaryRows(); quant=max(float(cloud.support[row])*1e-8,1e-12); lookup={}
                for j in order:
                    key=tuple(np.rint(directions[j]/1e-8).astype(np.int64))+(int(round(clearance[j]/quant)),)
                    peer=m[near[take[j]]].copy()
                    k=lookup.get(key)
                    if k is None:
                        lookup[key]=len(rows)
                        rows.append(dict(direction=directions[j].copy(),clearance=float(clearance[j]),
                            opening=float(opening[j]),normal=peer,segments=[endpoints[j].copy()],anchors=[anchors[j].copy()]))
                    else:
                        entry=rows[k]; entry['segments'].append(endpoints[j].copy()); entry['anchors'].append(anchors[j].copy())
                        if opening[j]<entry['opening']:
                            entry['opening']=float(opening[j]); entry['normal']=peer
                    _edge_merge_density(rows[lookup[key]],peaks,j,peer_positions[peers[j]])
                found[row]=rows
                if work is not None: rows.work_key=keys[local]
    return found,tested


def _geometric_local_candidates(rows,position,normal,base_position,rot,scale):
    """World finite boundary segments -> original Gaussian's normalized UV frame."""
    directions=np.array([r['direction'] for r in rows])
    clearance=np.array([r['clearance'] for r in rows])+directions@(base_position-position)
    u,v=_render_axes(np.asarray(rot).reshape(1,4)); intervals=[]
    planes=world_clip_to_local(directions,clearance,rot,scale,axes=(u,v))
    for row,plane in zip(rows,planes):
        length=float(np.linalg.norm(plane[:2])); tangent=np.array([-plane[1],plane[0]])/max(length,1e-30)
        ends=np.asarray(row['segments']).reshape(-1,2,3)-base_position
        uv=np.stack((ends@u[0]/scale[0],ends@v[0]/scale[1]),axis=-1)
        values=uv@tangent
        intervals.append(np.sort(values,axis=1))
    return planes,intervals


_clip_axis=(np.arange(32,dtype=float)+.5)*(6./32.)-3.


_clip_x,_clip_y=np.meshgrid(_clip_axis,_clip_axis,indexing='ij')


_CLIP_UV=np.column_stack((_clip_x.ravel(),_clip_y.ravel()))


_CLIP_UV=_CLIP_UV[np.sum(_CLIP_UV**2,axis=1)<=9.]


_CLIP_WEIGHTS=np.exp(-.5*np.sum(_CLIP_UV**2,axis=1))


_CLIP_WEIGHTS/=_CLIP_WEIGHTS.sum()


def _clip_pair_topology(rows,planes,segments,position,scale):
    """Infer local AND/OR joins from finite boundary-face continuation.

    A notch's two ridge faces continue OUTSIDE each other's retained side.
    An exterior convex corner continues INSIDE. Ambiguous/mixed evidence keeps
    the existing AND approximation and is reported, never silently called OR.
    No cutter, SDF/history or clipped/render-only children are queried.
    """
    count=len(rows); concave=np.zeros((count,count),bool)
    uncertain=np.zeros_like(concave)
    if count<2: return concave,uncertain
    planes=np.asarray(planes,float)
    norms=np.linalg.norm(planes[:,:2],axis=1)
    unit=planes/np.maximum(norms[:,None],1e-30)
    a,b,c=unit.T
    det=a[:,None]*b[None,:]-b[:,None]*a[None,:]
    nonparallel=np.abs(det)>1e-5
    x=np.divide(b[:,None]*c[None,:]-c[:,None]*b[None,:],det,
                out=np.zeros_like(det),where=nonparallel)
    y=np.divide(c[:,None]*a[None,:]-a[:,None]*c[None,:],det,
                out=np.zeros_like(det),where=nonparallel)
    valid=nonparallel&(x*x+y*y<9.-1e-6)
    tol=1e-5
    for i,intervals in enumerate(segments):
        values=-b[i]*x[i]+a[i]*y[i]
        intervals=np.asarray(intervals,float).reshape(-1,2)
        if len(intervals):
            on=np.any((values[None,:]>=intervals[:,0,None]-tol)&
                      (values[None,:]<=intervals[:,1,None]+tol),axis=0)
            valid[i]&=on
        else: valid[i]=False
    valid&=valid.T
    if not np.any(valid): return concave,uncertain
    directions=np.asarray([r['direction'] for r in rows])
    clearance=np.asarray([r['clearance'] for r in rows])
    inside=np.zeros_like(valid); outside=np.zeros_like(valid)
    sigma=max(float(np.max(scale[:2])),1e-12)
    for i,row in enumerate(rows):
        js=np.flatnonzero(valid[i])
        if not len(js): continue
        anchors=np.asarray(row['anchors'],float).reshape(-1,3)
        ends=np.asarray(row['segments'],float).reshape(-1,2,3)
        delta=anchors-position
        weights=np.linalg.norm(ends[:,1]-ends[:,0],axis=1)
        weights*=np.exp(-.5*np.sum(delta*delta,axis=1)/(sigma*sigma))
        side=delta@directions[js].T+clearance[js]
        eps=1e-5*sigma
        pos=np.sum(weights[:,None]*(side>eps),axis=0)
        neg=np.sum(weights[:,None]*(side < -eps),axis=0)
        total=pos+neg
        reliable=total>1e-12*sigma
        inside[i,js]=reliable&(pos>=.90*total)
        outside[i,js]=reliable&(neg>=.90*total)
    concave=valid&outside&outside.T
    convex=valid&inside&inside.T
    uncertain=valid&~(concave|convex)
    return concave,uncertain


def _disjoint_or_clip_values(planes):
    """A OR B -> A, (strictly outside A) AND B, without moving the GS.

    The angle/split API only has inclusive >=. Separate the complementary plane
    by a tiny normalized-UV guard, including Float32/trigonometric rounding.
    This intentionally leaves a numerical-width seam rather than overlapping.
    The non-overlap bound covers |UV| <= 16, beyond meaningful Gaussian support.
    Exact gap-free half-open ownership would require a shader '<' operator.
    """
    planes=np.asarray(planes,float)
    if planes.shape!=(2,3): raise ValueError('OR partition requires exactly two boundaries')
    first=local_planes_to_clip_values(planes[:1])
    second_b=local_planes_to_clip_values(planes[1:])
    theta=float(first[0]); split=float(first[1])
    opposite=np.float32(theta+np.pi if theta<0 else theta-np.pi)
    if opposite>=np.float32(np.pi): opposite=-np.float32(np.pi)
    n=np.array([np.cos(first[0]),np.sin(first[0])],np.float32).astype(float)
    reversed_n=np.array([np.cos(opposite),np.sin(opposite)],np.float32).astype(float)
    radius=16.
    guard=radius*np.linalg.norm(n+reversed_n)+32.*np.finfo(np.float32).eps*(1.+abs(split)+radius)
    threshold=np.nextafter(np.float32(-split+guard),np.float32(np.inf))
    second=np.array([opposite,threshold,second_b[0],second_b[1]],np.float32)
    return first,second,float(threshold)+split


def select_important_clip_planes(planes,*,ridge_mask,opening_angles,
                                 tangent_scales=(1.,1.),position_floor=.10,
                                 diversity_floor=.15,boundary_samples=16,boundary_segments=None,concave_pairs=None,density_log_peaks=None,evaluation_radius=2.0,similar_edge_angle_deg=15.):
    """Peak own-alpha x peer-alpha on finite segments, then diverse second.

    Segment length is diagnostic only. Unselected half-planes never trim the
    score domain. density_log_peaks must come from actual paired Gaussians.
    """
    planes=np.asarray(planes,dtype=np.float64)
    if planes.ndim!=2 or planes.shape[1]!=3:
        raise ValueError('Expected (K,3) clip planes')
    count=len(planes)
    concave=np.zeros((count,count),bool) if concave_pairs is None else np.asarray(concave_pairs,bool)
    if concave.shape!=(count,count) or not np.array_equal(concave,concave.T) or np.any(np.diag(concave)):
        raise ValueError('Concave joins must be a symmetric boundary-pair matrix with zero diagonal')
    if boundary_segments is not None and len(boundary_segments)!=count:
        raise ValueError("One finite segment collection per candidate is required")
    ridge=np.asarray(ridge_mask,dtype=bool).reshape(-1)
    opening=np.asarray(opening_angles,dtype=float).reshape(-1)
    scales=np.asarray(tangent_scales,dtype=float).reshape(2)
    if len(ridge)!=count or len(opening)!=count:
        raise ValueError('One ridge classification and opening angle per boundary is required')
    if not np.isfinite(opening).all() or np.any((opening<0)|(opening>np.pi)):
        raise ValueError('Opening angles must be finite radians in [0,pi]')
    if not np.isfinite(scales).all() or np.any(scales<=0):
        raise ValueError('Tangent scales must be positive and finite')
    if not (0<=position_floor<=1 and 0<=diversity_floor<=1) or boundary_samples<1:
        raise ValueError('Clip score floors must be in [0,1], with at least one sample')
    with np.errstate(over='ignore',invalid='ignore'):
        uploaded=planes.astype(np.float32).astype(np.float64)
    if not np.isfinite(uploaded).all():
        raise ValueError('Candidate planes must remain finite in Float32')
    norm=np.linalg.norm(uploaded[:,:2],axis=1)
    valid=norm>1e-30
    unit=uploaded/np.maximum(norm[:,None],1e-30)
    canonical=np.lexsort((unit[:,2],unit[:,1],unit[:,0]))
    if density_log_peaks is None:
        raise ValueError('Actual paired Gaussian log-density peaks are required')
    log_peak=np.asarray(density_log_peaks,float)
    if log_peak.shape!=(count,) or np.any(np.isnan(log_peak)) or np.any(log_peak>1e-12):
        raise ValueError('One log-density peak <= 0 per candidate is required; -inf means no common range')
    effective=np.zeros(count)
    sharpness=np.where(ridge,np.cos(.5*opening)**2,0.)
    intervals=np.zeros((count,2))
    metric=scales/np.sqrt(scales[0]*scales[1])
    tangents=np.column_stack((-unit[:,1],unit[:,0]))
    half=np.sqrt(np.maximum(0.,9.-unit[:,2]**2))
    active=ridge&valid&(np.abs(unit[:,2])<3.)
    finite_segments=[]
    for i in np.flatnonzero(active):
        pieces=np.array([[-half[i],half[i]]]) if boundary_segments is None else np.asarray(boundary_segments[i],float).reshape(-1,2)
        # Segments already came from BOTH finite disks. Do not occlude them
        # with unselected infinite half-planes or weight by their length.
        pieces=pieces[pieces[:,1]>pieces[:,0]+1e-10]
        if not len(pieces): continue
        pieces=pieces[np.lexsort((pieces[:,1],pieces[:,0]))]
        merged=[]
        for a,b in pieces:
            if merged and a<=merged[-1][1]+1e-10: merged[-1][1]=max(merged[-1][1],b)
            else: merged.append([float(a),float(b)])
        pieces=np.asarray(merged)
        intervals[i]=(pieces[0,0],pieces[-1,1])
        effective[i]=float(np.sum(pieces[:,1]-pieces[:,0]))*np.linalg.norm(tangents[i]*metric)
        finite_segments.extend((int(i),float(a),float(b)) for a,b in pieces)
    active&=(effective>0)&(sharpness>0)&np.isfinite(log_peak)
    log_importance=np.full(count,-np.inf)
    log_importance[active]=log_peak[active]+np.log(sharpness[active])
    density_peak=np.exp(log_peak)  # diagnostics only; ranking stays in log space
    importance=np.exp(log_importance)
    # Exact duplicate lines must not consume the second slot, even if generated
    # by two separate faces. Keep the strongest relation, then canonical order.
    eligible=active.copy()
    ranked=sorted(canonical,key=lambda i:(-log_importance[i],float(unit[i,0]),float(unit[i,1]),float(unit[i,2])))
    unique=[]
    for i in ranked:
        if not eligible[i]: continue
        if any(np.max(np.abs(unit[i]-unit[j]))<=1e-8 for j in unique): eligible[i]=False
        else: unique.append(i)
    group_representative=np.where(eligible,np.arange(count),-1).astype(np.int32)
    group_log_importance=np.where(eligible,log_importance,-np.inf)
    if similar_edge_angle_deg!=0. and np.count_nonzero(eligible)>1:
        group_representative,group_log_importance=_edge_group_same_boundary(
            uploaded,eligible,log_importance,scales,evaluation_radius,similar_edge_angle_deg,concave)
    similar_suppressed=eligible&(group_representative!=np.arange(count))
    eligible&=~similar_suppressed
    group_importance=np.exp(group_log_importance)
    selected=[]; selected_scores=[]
    direction_difference=np.zeros(count); position_difference=np.zeros(count)
    additional_effect=np.zeros(count); diversity=np.zeros(count); second_score=np.zeros(count)
    rejects=None
    def choose(score,allowed):
        best=float(np.max(score[allowed]))
        tol=1e-12*max(1.,abs(best))
        return int(next(i for i in canonical if allowed[i] and score[i]>=best-tol))
    if np.any(eligible):
        first=choose(group_log_importance,eligible); selected.append(first); selected_scores.append(float(group_importance[first]))
        remaining=eligible.copy(); remaining[first]=False
        if np.any(remaining):
            # Approximate area is used only for the SECOND edge's incremental
            # effect and diagnostics, never as the first edge's importance.
            # Align quadrature to the first boundary. A fixed UV grid made
            # otherwise equivalent circular-rim neighbours choose different
            # second edges solely because their local frame was rotated.
            first_normal=unit[first,:2]
            first_tangent=np.array([-first_normal[1],first_normal[0]])
            sample_uv=(_CLIP_UV[:,:1]*first_normal+
                       _CLIP_UV[:,1:]*first_tangent)
            rejects=(sample_uv@uploaded[:,:2].T+uploaded[:,2])<0.
            area_weights=position_floor/len(_CLIP_UV)+(1.-position_floor)*_CLIP_WEIGHTS
            rejected_mass=np.sum(rejects*area_weights[:,None],axis=0)
            new_mass=np.sum(rejects*(area_weights*~rejects[:,first])[:,None],axis=0)
            additional_effect=np.divide(new_mass,rejected_mass,out=np.zeros(count),where=rejected_mass>1e-15)
            # For OR, B contributes restored area outside A, not an extra cut.
            restores=np.sum((~rejects)*(area_weights*rejects[:,first])[:,None],axis=0)
            if rejected_mass[first]>1e-15:
                additional_effect[concave[first]]=restores[concave[first]]/rejected_mass[first]
            # Undo anisotropic Gaussian scaling for PHYSICAL direction/position
            # differences. Sharpness also uses world face normals, not UV angles.
            physical=uploaded[:,:2]/metric
            physical_norm=np.linalg.norm(physical,axis=1)
            physical_unit=physical/np.maximum(physical_norm[:,None],1e-30)
            direction_difference=np.clip(.5*(1.-physical_unit@physical_unit[first]),0.,1.)
            feet=-(uploaded[:,2]/np.maximum(physical_norm,1e-30))[:,None]*physical_unit
            position_difference=np.clip(np.linalg.norm(feet-feet[first],axis=1)/(6.*max(metric)),0.,1.)
            diversity=additional_effect*(.5+.5*np.maximum(direction_difference,position_difference))
            second_score=group_importance*(diversity_floor+(1.-diversity_floor)*diversity)
            with np.errstate(divide='ignore'):
                second_log=group_log_importance+np.log(diversity_floor+(1.-diversity_floor)*diversity)
            if not np.any(np.isfinite(second_log[remaining])): second_log=group_log_importance
            second=choose(second_log,remaining)
            selected.append(second); selected_scores.append(float(second_score[second]))
    join_or=len(selected)==2 and bool(concave[selected[0],selected[1]])
    if len(selected)<count:
        if rejects is None: rejects=(_CLIP_UV@uploaded[:,:2].T+uploaded[:,2])<0.
        covered=(np.all(rejects[:,selected],axis=1) if join_or else np.any(rejects[:,selected],axis=1)) if selected else np.zeros(len(_CLIP_UV),bool)
        unselected=np.ones(count,bool); unselected[selected]=False
        omitted=float(_CLIP_WEIGHTS[np.any(rejects[:,unselected],axis=1)&~covered].sum())
    else: omitted=0.
    return dict(planes=planes.copy(),selected_indices=np.asarray(selected,dtype=np.int32),
        concave_pairs=concave.copy(),join_or=join_or,
        importance=importance,marginal_importance=np.asarray(selected_scores),omitted_mass=omitted,occlusion_fallback=False,
        ridge_mask=ridge.copy(),opening_degrees=np.degrees(opening),sharpness=sharpness,
        density_peak=density_peak,log_density_peak=log_peak.copy(),log_importance=log_importance,
        effective_length=effective,eligible=eligible,score_method='peak_alpha_product',
        similar_direction_suppressed=similar_suppressed,similar_edge_angle_deg=float(similar_edge_angle_deg),
        group_representative=group_representative,group_importance=group_importance,group_log_importance=group_log_importance,
        boundary_intervals=intervals,finite_segments=np.asarray(finite_segments,dtype=float).reshape(-1,3),
        direction_difference=direction_difference,
        position_difference=position_difference,additional_effect=additional_effect,
        diversity=diversity,second_score=second_score)


def _pack_native_clip(cloud,material,settings,*,reference_store=None):
    from cutsim.core.chunked_cloud import ChunkedCloud
    if isinstance(cloud,ChunkedCloud):cloud=cloud.dense_view()
    # The CPU union is material geometry, not a native Top-2 drawing domain.
    # Its immutable projection was produced with the shared cut records.
    from cutsim.rendering.strip_projection import drawing_cloud
    surface_domains=cloud.sweep_domains
    cloud=drawing_cloud(cloud)
    # Effective ridge and valley boundaries are eligible, at most two per parent.
    # Do not reject planar Gaussians for disagreeing with the curved material SDF.
    out=NativeClipArrays(_pack_gaussian_attributes_legacy(cloud,material,
        replace(settings,half_clip=False,half_double_edge_split=False)))
    out.gs_density=getattr(material,'_gs_density',1.)
    out.draw_subdivision_floor_scale=settings.draw_subdivision_floor_scale
    out.pop('boundary_split')
    out['surface_id']=surface_ids(cloud,material)
    count=len(cloud.positions)
    for name,value in zip(CLIP_ATTRIBUTE_NAMES,(4.,0.,4.,0.)):
        out[name]=np.full((count,1),value,np.float32)
    out['clip_edge_count']=np.zeros(count,np.int32)
    # Zero-valued compatibility fields for existing caches/performance-log readers.
    out['clip_pending']=np.zeros(count,bool); out['clip_reason']=np.zeros(count,np.int8)
    out['clip_empty']=np.zeros(count,bool); out['clip_sweep_top2']=np.zeros(count,bool)
    out['edge_case']=np.zeros(count,np.int16); out.pending={}; out.approximated={}
    out['clip_candidate_count']=np.zeros(count,np.int32)
    # Compatibility key: counts all ridge + valley candidates.
    out['clip_ridge_count']=np.zeros(count,np.int32)
    out['clip_eligible_count']=np.zeros(count,np.int32)
    out['clip_occlusion_fallback']=np.zeros(count,bool)
    out['clip_concave']=np.zeros(count,bool)
    out['clip_join_uncertain']=np.zeros(count,bool)
    out['clip_second_values']=np.tile(np.array([4.,0.,4.,0.],np.float32),(count,1))
    out['clip_partition_guard']=np.zeros(count,np.float32)
    out['clip_selected_indices']=np.full((count,2),-1,np.int32)
    out['clip_selection_scores']=np.zeros((count,2),np.float32)
    out['clip_approximated']=np.zeros(count,bool)
    out['clip_omitted_mass']=np.zeros(count,np.float32)
    if settings.half_clip and count:
        active=np.fromiter((i for i,domain in enumerate(cloud.sweep_domains) if domain is None),dtype=np.int64)
        candidates,_=_native_boundaries(cloud,material,out,settings,reference_store=reference_store,active_rows=active,share_work=True,surface_domains=surface_domains)
        for i,rows in enumerate(candidates):
            if not rows or cloud.sweep_domains[i] is not None: continue
            work=getattr(material,'_boundary_work',None)
            selection=getattr(rows,'selection',None) if work is not None else None
            if selection is None:
                planes,segments=_geometric_local_candidates(rows,cloud.positions[i],cloud.normals[i],
                    out['positions'][i],out['rot'][i],out['scale'][i])
                if not len(planes): continue
                ridge=np.ones(len(planes),dtype=bool)
                opening=np.array([r['opening'] for r in rows])
                concave,uncertain=_clip_pair_topology(rows,planes,segments,cloud.positions[i],out['scale'][i])
                choice=select_important_clip_planes(planes,ridge_mask=ridge,opening_angles=opening,
                    tangent_scales=out['scale'][i,:2],position_floor=settings.clip_position_floor,
                    diversity_floor=settings.clip_diversity_floor,boundary_samples=settings.clip_boundary_samples,
                    boundary_segments=segments,concave_pairs=concave,
                    density_log_peaks=np.array([r['log_density_peak'] for r in rows]),
                    evaluation_radius=getattr(settings,'intersection_sigma',3.),
                    similar_edge_angle_deg=getattr(settings,'similar_edge_angle_deg',15.))
                adjacent_normals=np.asarray([r['normal'] for r in rows],dtype=float)
                if work is not None and getattr(rows,'work_key',None) is not None:
                    work.put(rows.work_key,PreparedBoundary((planes,uncertain,choice),adjacent_normals))
            else:
                planes,uncertain,choice=selection
                ridge=np.ones(len(planes),dtype=bool)
                adjacent_normals=rows.normals
                work.selection_hits+=1
            number=len(planes); out['clip_candidate_count'][i]=number
            selected=choice['selected_indices']; applied=planes[selected]
            out['clip_ridge_count'][i]=int(np.count_nonzero(ridge))
            out['clip_eligible_count'][i]=int(np.count_nonzero(choice['eligible']))
            out['clip_selected_indices'][i,:len(selected)]=selected
            out['clip_selection_scores'][i,:len(selected)]=choice['marginal_importance']
            out['clip_approximated'][i]=len(selected)<number
            out['clip_omitted_mass'][i]=choice['omitted_mass']
            out['clip_occlusion_fallback'][i]=choice['occlusion_fallback']
            out['clip_concave'][i]=choice['join_or']
            out['clip_join_uncertain'][i]=len(selected)==2 and uncertain[selected[0],selected[1]]
            if number>2 or len(selected)<number or choice['occlusion_fallback'] or choice['join_or'] or out['clip_join_uncertain'][i]:
                out.approximated[i]=dict(choice,selected_planes=applied.copy(),
                    adjacent_normals=adjacent_normals.copy(),join_uncertain=uncertain.copy())
            out['clip_edge_count'][i]=len(applied)
            if len(applied):
                if choice['join_or']:
                    values,second,guard=_disjoint_or_clip_values(applied)
                    out['clip_second_values'][i]=second; out['clip_partition_guard'][i]=guard
                else: values=local_planes_to_clip_values(applied)
                for name,value in zip(CLIP_ATTRIBUTE_NAMES,values): out[name][i,0]=value
    _apply_inherited_clip(out,cloud)
    from cutsim.rendering.stock_clipping import bound_drawing_cloud
    if settings.clip_stock_bounds:
        cloud=bound_drawing_cloud(out,cloud,material)
    from cutsim.rendering.cap_junctions import bound_cap_junctions
    cloud=bound_cap_junctions(cloud,out,material)
    _apply_sweep_domains(out,cloud)
    _native_clip_stats(out,material)
    return out


def _resolve_pending_clip_draws(out):
    """Keep geometric records intact; prepare bounded drawing-only replacements."""
    if getattr(out,'_clip_draws_resolved',False):return
    metadata=getattr(out,'approximated',None) or {}
    if not metadata:return
    from cutsim.rendering.parent_parallel import resolve_parallel
    if resolve_parallel(out,metadata):return
    for i in np.flatnonzero(out['clip_pending'] & ~out['clip_empty']):
        diagnostic=metadata.get(int(i),{})
        if 'pending_fallback' in diagnostic:continue
        regions=diagnostic.get('fallback_regions_local')
        if regions is None:continue
        record={name:value[i] for name,value in out.items()}
        finite=diagnostic.get('finite_strip_domain',False)
        stock=diagnostic.get('stock_planes_local')
        from cutsim.geometry.sampling_density import DensityPolicy
        policy=DensityPolicy(getattr(out,'gs_density',1.))
        minimum=policy.draw_floor(regions)*getattr(out,'draw_subdivision_floor_scale',1.)
        from cutsim.rendering.exact_draw_cache import fallback_cache, array_key
        from cutsim.rendering.gs_schema import GS_FIELDS
        from cutsim.rendering.fallback_kernels import subdivision_policy
        from cutsim.rendering.coverage_cuda import backend_key
        key=(subdivision_policy(),backend_key(),finite, minimum, tuple((name,array_key(record[name])) for name in GS_FIELDS+('coverage_logit',) if name in record),
             tuple(array_key(region) for region in regions),None if stock is None else array_key(stock))
        cached=fallback_cache.get(key)
        if cached is not None:
            diagnostic['pending_fallback']=cached
            continue
        limit=2048
        while True:
            fallback=build_pending_fallback(record,regions,prune=True,
                base_depth=0 if finite else 3,max_children=limit,preserve_simple_regions=finite,
                min_child_sigma=minimum,stock_planes=stock)
            if not finite or 'budget' not in fallback['incomplete_reasons']:break
            # Repeat a bounded subdivision with more storage. Never publish a
            # partially processed strip domain as a completed render record.
            limit*=2
        diagnostic['pending_fallback']=fallback
        fallback_cache.put(key,fallback)


def _native_clip_stats(out,material,**local):
    if getattr(out,'_immutable_chunks',False):
        stats=out.native_stats()
        stats.update(local)
        material.last_edge_split_stats=stats
        return
    counts=out['clip_edge_count']; pending=unresolved_clip_mask(out); ready=~(out['clip_pending']|out['clip_empty'])
    abandoned=abandoned_clip_mask(out)
    size_approximate=size_approximation_mask(out)
    fallbacks=pending_fallbacks(out)
    fallback_children=sum(entry['children'] for entry in fallbacks.values())
    fallback_before=sum(entry.get('children_before_prune',entry['children']) for entry in fallbacks.values())
    stats=_edge_stats_empty()
    stats.update(clip_empty=int(out['clip_empty'].sum()),clip_sweep_top2=int(out['clip_sweep_top2'].sum()),clip_native=1,clip_zero=int(np.count_nonzero(ready&(counts==0))),
        clip_one=int(np.count_nonzero(ready&(counts==1))),clip_two=int(np.count_nonzero(ready&(counts==2))),
        clip_pending=int(np.count_nonzero(pending)),clip_over_two=int(np.count_nonzero(out['clip_reason']==1)),
        clip_non_intersection=int(np.count_nonzero(out['clip_reason']==2)),
        clip_uncertain=0,clip_fixed_top_two=1,clip_ridge_selection=1,clip_intersection_rejection=0,edge_recomputed=0,edge_reused=0)
    approximate=out['clip_approximated']
    stats.update(clip_approximated=int(np.count_nonzero(approximate)),
        clip_candidates_over_two=int(np.count_nonzero(out['clip_candidate_count']>2)),
        clip_omitted_edges=int(np.sum(out['clip_candidate_count']-counts)),
        clip_ridge_candidates=int(np.sum(out['clip_ridge_count'])),
        clip_eligible_candidates=int(np.sum(out['clip_eligible_count'])),
        clip_finite_fallback=int(np.count_nonzero(out['clip_occlusion_fallback'])),
        clip_non_ridge_candidates=int(np.sum(out['clip_candidate_count']-out['clip_ridge_count'])),
        clip_ridge_over_two=int(np.count_nonzero(out['clip_eligible_count']>2)),
        clip_omitted_mass_max=float(np.max(out['clip_omitted_mass'],initial=0.)))
    stats.update(clip_concave_parents=int(np.count_nonzero(out['clip_concave'])),
        clip_draw_count=int(ready.sum())+int(np.count_nonzero(out['clip_concave']&ready))+fallback_children,
        clip_fallback_parents=sum(entry['complete'] for entry in fallbacks.values()),
        clip_fallback_children=fallback_children,
        clip_fallback_children_before_prune=fallback_before,
        clip_fallback_pruned=fallback_before-fallback_children,
        clip_fallback_prune_ms=sum(entry.get('pruning',{}).get('milliseconds',0.) for entry in fallbacks.values()),
        clip_fallback_size_stopped=sum(entry.get('subdivision',{}).get('size_stopped',0) for entry in fallbacks.values()),
        clip_fallback_size_discarded=sum(entry.get('subdivision',{}).get('size_discarded',0) for entry in fallbacks.values()),
        clip_fallback_abandoned=int(np.count_nonzero(abandoned)),
        clip_fallback_top_two_parents=int(np.count_nonzero(size_approximate)),
        clip_fallback_top_two_children=sum(entry.get('subdivision',{}).get('top_two_children',0) for entry in fallbacks.values()),
        clip_marker_pending=int(np.count_nonzero(pending & ~(abandoned|size_approximate))),
        clip_join_uncertain=int(np.count_nonzero(out['clip_join_uncertain'])),
        clip_partition_guard_max=float(np.max(out['clip_partition_guard'],initial=0.)))
    stats.update(local); material.last_edge_split_stats=stats


def _clip_world_regions(arr):
    """Actual uploaded, disjoint draw branches -> immutable world halfplanes."""
    count=len(arr['positions']); u,v=_render_axes(arr['rot'])
    result=np.zeros((count,2,2,4)); result[:,:,:,3]=-1.
    values=np.stack((np.column_stack([arr[k][:,0] for k in CLIP_ATTRIBUTE_NAMES]),
                     arr['clip_second_values']),axis=1)
    for branch in (0,1):
        active=(~(arr['clip_pending']|arr.get('clip_empty',False))) & ((branch==0)|arr['clip_concave'])
        for edge in (0,1):
            angle=values[:,branch,edge*2]; split=values[:,branch,edge*2+1]
            normal=(np.cos(angle)[:,None]*u/arr['scale'][:,0,None]+
                    np.sin(angle)[:,None]*v/arr['scale'][:,1,None])
            plane=np.column_stack((normal,-np.sum(normal*arr['positions'],axis=1)-split))
            plane[angle==4.]=[0.,0.,0.,1.]
            result[active,branch,edge]=plane[active]
    return result


def _apply_inherited_clip(arr,cloud):
    """Intersect inherited and new disjoint branches without discarding constraints.

    Up to two boundaries per branch and two disjoint branches are representable.
    More complex intersections are kept in geometry, but hidden and diagnosed.
    """
    inherited=cloud.inherited_clip
    ids=np.flatnonzero(np.any(inherited[:,:,:,:3]!=0.,axis=(1,2,3)))
    if not len(ids):return
    u,v=_render_axes(arr['rot'][ids])
    for row,i in enumerate(ids):
        local=[]
        for region in inherited[i]:
            planes=np.column_stack((region[:,:3]@u[row]*arr['scale'][i,0],
                                    region[:,:3]@v[row]*arr['scale'][i,1],
                                    region[:,:3]@arr['positions'][i]+region[:,3]))
            local.append(planes)
        values=[np.array([arr[k][i,0] for k in CLIP_ATTRIBUTE_NAMES])]
        if arr['clip_concave'][i]:values.append(arr['clip_second_values'][i])
        current=[]
        for value in values:
            current.append(np.array([[0.,0.,1.] if value[j]==4. else
                [np.cos(value[j]),np.sin(value[j]),-value[j+1]] for j in (0,2)]))
        combined=[]; full_regions=[]; unsupported=False
        for old in local:
            for new in current:
                constraints=[]; empty=False
                for plane in np.concatenate((old,new)):
                    length=np.linalg.norm(plane[:2])
                    if length<1e-12:
                        if plane[2]<0:empty=True;break
                        continue
                    plane=plane/length
                    # Only parallel same-facing lines can be merged globally.
                    match=next((j for j,q in enumerate(constraints) if np.linalg.norm(plane[:2]-q[:2])<1e-7),None)
                    if match is None:constraints.append(plane)
                    elif plane[2]<constraints[match][2]:constraints[match]=plane
                if empty:continue
                # Exact opposing parallel inequalities may prove the region empty.
                if any(np.linalg.norm(a[:2]+b[:2])<1e-7 and a[2]+b[2]<-1e-7
                       for j,a in enumerate(constraints) for b in constraints[j+1:]):continue
                full_regions.append(np.asarray(constraints).reshape(-1,3))
                if len(constraints)>2:unsupported=True;continue
                combined.append(local_planes_to_clip_values(np.asarray(constraints).reshape(-1,3)))
        if unsupported or len(combined)>2 or not combined:
            arr['clip_pending'][i]=True;arr['clip_reason'][i]=4
            arr['clip_concave'][i]=False
            arr.approximated.setdefault(int(i),{}).update(inherited_unresolved=True,
                fallback_regions_local=tuple(full_regions))
            continue
        arr['clip_concave'][i]=len(combined)==2
        for k,value in zip(CLIP_ATTRIBUTE_NAMES,combined[0]):arr[k][i,0]=value
        arr['clip_second_values'][i]=combined[1] if len(combined)==2 else [4.,0.,4.,0.]
        arr['clip_edge_count'][i]=max(sum(value[j]!=4. for j in (0,2)) for value in combined)


def _sweep_support(sweep,normals):
    """Support of cutter volume (or union hull), so retained halfspace is safe."""
    if hasattr(sweep, 'support'):
        return sweep.support(normals)
    n=np.asarray(normals,float)
    xy=np.maximum(n[...,:2]@sweep.a_xy,n[...,:2]@sweep.b_xy)
    radial=sweep.radius*np.linalg.norm(n[...,:2],axis=-1)
    if isinstance(sweep,FlatSweep):
        return xy+radial+np.maximum(n[...,2]*sweep.z_lo,n[...,2]*sweep.z_hi)
    if isinstance(sweep,BallSweep):
        ball=xy+n[...,2]*sweep.center_z+sweep.radius*np.linalg.norm(n,axis=-1)
        upper=xy+radial+np.maximum(n[...,2]*sweep.center_z,n[...,2]*sweep.z_hi)
        return np.maximum(ball,upper)
    raise TypeError('Sweep boundary requires a FlatSweep or BallSweep')


def _sweep_display_rows(index,cloud,sweep):
    lo,hi=sweep.world_aabb()
    if isinstance(sweep,BallSweep):
        hi=hi.copy();hi[2]=max(hi[2],sweep.center_z+sweep.radius)
    rows=index.footprints.query(lo,hi,index.ids)
    # Conservative envelope also supports non-UI sigma_scale values up to 1.0.
    radius=3.*cloud.support[rows]*(1.+.2*np.clip(cloud.seam_weight[rows],0.,1.))
    return rows[np.asarray(sweep.sdf(cloud.positions[rows]))<=radius+1e-7]


def _reduce_world_region(planes):
    result=[]
    for raw in planes:
        plane=np.asarray(raw,float).copy();length=np.linalg.norm(plane[:3])
        if length<1e-12:
            if plane[3]<0:return None
            continue
        plane/=length
        match=next((j for j,q in enumerate(result) if np.linalg.norm(q[:3]-plane[:3])<1e-9),None)
        if match is None:result.append(plane)
        elif plane[3]<result[match][3]:result[match]=plane
    if any(np.linalg.norm(a[:3]+b[:3])<1e-9 and a[3]+b[3]<-1e-8
           for j,a in enumerate(result) for b in result[j+1:]):return None
    return np.asarray(result).reshape(-1,4)


def _curve_nearest_intersections(source,target,centres,normals,sigma):
    """Project onto the common implicit curve, then minimize distance along it.

    Three starts detect competing curve branches. No GS neighbour sampling or
    candidate-angle quantization determines the accepted tangent plane.
    """
    p=np.asarray(centres,float);sigma=np.asarray(sigma,float)
    _,gt=target.sdf_gradient(p)
    direction=gt-np.sum(gt*normals,axis=1)[:,None]*normals
    direction/=np.maximum(np.linalg.norm(direction,axis=1)[:,None],1e-12)
    x=p[:,None,:]+np.array([0.,-2.,2.])[None,:,None]*sigma[:,None,None]*direction[:,None,:]
    origin=np.repeat(p,3,axis=0);scale=np.repeat(sigma,3);x=x.reshape(-1,3)
    valid=np.ones(len(x),bool)
    for _ in range(18):
        f,a=source.sdf_gradient(x);g,b=target.sdf_gradient(x)
        dot=np.sum(a*b,axis=1);det=1.-dot*dot
        valid &= det>.03
        divisor=np.maximum(det,.03)
        correction=((f-dot*g)/divisor)[:,None]*a+((g-dot*f)/divisor)[:,None]*b
        tangent=np.cross(a,b);tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-12)
        slide=np.sum((origin-x)*tangent,axis=1)[:,None]*tangent
        step=-correction+slide
        step*=np.minimum(1.,scale/np.maximum(np.linalg.norm(step,axis=1),1e-12))[:,None]
        x+=step
    f,a=source.sdf_gradient(x);g,b=target.sdf_gradient(x)
    tangent=np.cross(a,b);tn=np.linalg.norm(tangent,axis=1)
    tangent/=np.maximum(tn[:,None],1e-12)
    distance=np.linalg.norm(x-origin,axis=1)
    valid &= (np.abs(f)<1e-7)&(np.abs(g)<1e-7)&(tn>.2)
    valid &= np.abs(np.sum((origin-x)*tangent,axis=1))<1e-5*np.maximum(scale,1.)
    valid &= (distance<2.*scale)&np.isfinite(x).all(axis=1)
    # Accept only smooth lower ball surfaces; reject cap/side transitions here.
    valid &= (x[:,2]<source.center_z-1e-5)&(x[:,2]<target.center_z-1e-5)
    distance=np.where(valid,distance,np.inf).reshape(-1,3)
    roots=x.reshape(-1,3,3);grad=b.reshape(-1,3,3)
    best=np.argmin(distance,axis=1);rows=np.arange(len(p));chosen=roots[rows,best]
    normal=grad[rows,best];ok=np.isfinite(distance[rows,best])
    # Distinct nearby roots mean a GS needs more than one local curve branch.
    distinct=np.linalg.norm(roots-chosen[:,None,:],axis=2)>.1*sigma[:,None]
    ok &= ~np.any(np.isfinite(distance)&distinct,axis=1)
    ok &= np.sum((-a.reshape(-1,3,3)[rows,best])*normals,axis=1)>.85
    offset=-_sweep_support(target,normal)
    ok &= np.abs(np.sum(normal*chosen,axis=1)+offset)<1e-6
    return np.column_stack((normal,offset)),ok,chosen


def _shared_curve_boundary_planes(cloud,target,rows,material):
    """Find generating surfaces through the existing LOCAL Sweep spatial index."""
    result={}
    if material is None or not isinstance(target,BallSweep) or not len(rows):return result
    p=cloud.positions[rows];normal=cloud.normals[rows]
    sigma=.75*cloud.support[rows]*(1.+.2*np.clip(cloud.seam_weight[rows],0.,1.))
    near=(np.abs(target.sdf(p))<2.*sigma)&(cloud.tool_ids[rows]>0)
    ids=np.flatnonzero(near)
    if not len(ids):return result
    # Batched AABB query covers just these update rows, never all history.
    reach=2.*sigma[ids,None]
    candidates=material.query_sweep_ids(np.min(p[ids]-reach,axis=0),np.max(p[ids]+reach,axis=0),original_boxes=False)
    best=np.full(len(rows),-1,int);error=np.full(len(rows),np.inf)
    for sid in candidates:
        source=material.sweeps[int(sid)]
        if not isinstance(source,BallSweep):continue
        if (source.radius==target.radius and source.center_z==target.center_z
            and np.array_equal(source.a_xy,target.a_xy) and np.array_equal(source.b_xy,target.b_xy)):continue
        same=ids[cloud.tool_ids[rows[ids]]==source.tool_id]
        if not len(same):continue
        f,g=source.sdf_gradient(p[same]);alignment=-np.sum(g*normal[same],axis=1)
        residual=np.abs(f)
        good=(residual<.12*cloud.support[rows[same]])&(alignment>.97)&(residual<error[same])
        chosen=same[good];best[chosen]=int(sid);error[chosen]=residual[good]
    for sid in np.unique(best[best>=0]):
        group=np.flatnonzero(best==sid)
        work=getattr(material,'_strip_work',None)
        args=(material.sweeps[int(sid)],target,p[group],normal[group],sigma[group])
        planes,ok,_=(_curve_nearest_intersections(*args) if work is None else
                     work.curve_boundaries(*args,calculate=_curve_nearest_intersections))
        for j in np.flatnonzero(ok):result[int(rows[group[j]])]=planes[j:j+1]
    return result


def _rebuild_added_curve_domains(cloud,material):
    """Use the same curve on the newly added side of a ball/ball crossing.

    Only additions with a verified nearby curve enter the authoritative Sweep
    path. Rebuild all local stock/Sweep constraints for that subset, so selecting
    a curve cannot accidentally discard a third tool or the workpiece boundary.
    """
    if not len(cloud.positions):return 0
    radius=2.*.75*cloud.support*(1.+.2*np.clip(cloud.seam_weight,0.,1.))
    lo=np.min(cloud.positions-radius[:,None],axis=0)
    hi=np.max(cloud.positions+radius[:,None],axis=0)
    rows=np.arange(len(cloud.positions));active=set()
    for sid in material.query_sweep_ids(lo,hi,original_boxes=False):
        target=material.sweeps[int(sid)]
        if isinstance(target,BallSweep):
            active.update(_shared_curve_boundary_planes(cloud,target,rows,material))
    if not active:return 0
    ids=np.array(sorted(active),dtype=np.int64);subset=_local_take(cloud,ids)
    _rebuild_child_sweep_domains(subset,material)
    cloud.inherited_clip[ids]=subset.inherited_clip
    cloud.sweep_domains[ids]=subset.sweep_domains
    return int(getattr(subset,'_curve_boundary_count',0))


def _flat_stock_face_tangent(sweep,position,normal,tool_id):
    """Closest lateral support preserves central mass on a horizontal stock GS.

    Finite ring quadrature can prefer two skew normals over this exact radial
    normal, leaving an artificial wedge outside a rounded cutter endpoint.
    This is a certified outside halfspace, not an offset or an enlarged GS.
    """
    if not isinstance(sweep,FlatSweep) or tool_id!=0 or abs(normal[2])<1.-1e-12:
        return None
    _,nearest=capsule2d_parameter(np.asarray(position)[None,:2],sweep.a_xy,sweep.b_xy)
    radial=position[:2]-nearest[0]
    length=np.linalg.norm(radial)
    if length<=1e-12:return None
    direction=np.r_[radial/length,0.]
    return np.r_[direction,-float(_sweep_support(sweep,direction))]


def _assign_sweep_domains(cloud,sweep,rows,parent_regions=None,material=None,boundary_planes=None):
    """Attach current cutter boundaries to survivors, regardless of Split status.

    Outside supporting halfspaces never contain cutter interior. Preserve the
    sampled pair and add the exact closest tangent for horizontal stock faces.
    Their OR is partitioned; native two-edge limits are handled by subdivision.
    """
    rows=np.asarray(rows,dtype=np.int64)
    if not len(rows):return rows
    # A cutter can touch a Gaussian's plane without entering it. Ring-gradient
    # samples need not include the exact supporting normal at that tangency;
    # their OR approximation would then carve a false gap around the contact.
    # Support in +/- the face normal certifies the ENTIRE plane outside the
    # cutter (or its enclosing union hull), independent of display size.
    normals=np.asarray(cloud.normals[rows],dtype=np.float64)
    normal_length=np.linalg.norm(normals,axis=1)
    offset=np.sum(normals*cloud.positions[rows],axis=1)
    outside=(normal_length>1e-12)&(np.maximum(offset-_sweep_support(sweep,normals),
                       -offset-_sweep_support(sweep,-normals))>=-1e-9*normal_length)
    if parent_regions is not None:
        parent_regions=np.asarray(parent_regions)[~outside]
    rows=rows[~outside]
    if not len(rows):return rows
    missing=rows if boundary_planes is None else np.asarray([i for i in rows if int(i) not in boundary_planes],np.int64)
    shared_planes=_shared_curve_boundary_planes(cloud,sweep,missing,material) if len(missing) else {}
    if boundary_planes:
        shared_planes.update(boundary_planes)
    cloud._curve_boundary_count=getattr(cloud,'_curve_boundary_count',0)+len(shared_planes)
    normals=cloud.normals[rows];u,v=tangent_frames(normals)
    sigma=cloud.support[rows]*.75*(1.+.2*np.clip(cloud.seam_weight[rows],0.,1.))
    theta=np.arange(16)*2.*np.pi/16.;ring=np.column_stack((np.cos(theta),np.sin(theta)))
    uv=np.concatenate((np.zeros((1,2)),ring,2.*ring,4.*ring))
    points=cloud.positions[rows,None,:]+sigma[:,None,None]*(uv[None,:,0,None]*u[:,None,:]+uv[None,:,1,None]*v[:,None,:])
    _,gradient=sweep.sdf_gradient(points.reshape(-1,3));gradient=gradient.reshape(len(rows),len(uv),3)
    weights=np.exp(-.5*np.sum(uv*uv,axis=1));changed=[]
    for rr,i in enumerate(rows):
        n=gradient[rr];length=np.linalg.norm(n,axis=1);n=n[length>1e-10]/length[length>1e-10,None]
        n=np.unique(np.round(n,12),axis=0)
        if not len(n):continue
        planes=shared_planes.get(int(i))
        if planes is None:planes=np.column_stack((n,-_sweep_support(sweep,n)))
        # Plane crossing the full possible display disk is the relevant test.
        reach=3.*cloud.support[i]*(1.+.2*np.clip(cloud.seam_weight[i],0.,1.))
        center=planes[:,:3]@cloud.positions[i]+planes[:,3]
        tangential=np.sqrt((planes[:,:3]@u[rr])**2+(planes[:,:3]@v[rr])**2)
        if np.any(center-reach*tangential>=-1e-9):continue # whole disk outside cutter
        previous=cloud.sweep_domains[i]
        base=(tuple(parent_regions[rr]) if parent_regions is not None else tuple(cloud.inherited_clip[i])) if previous is None else previous
        valid=np.zeros(len(uv),bool)
        for region in base:
            valid|=np.all(points[rr]@region[:,:3].T+region[:,3]>=-1e-8,axis=1)
        keep=points[rr]@planes[:,:3].T+planes[:,3]>=0.
        score=(weights*valid)@keep
        first=int(np.argmax(score));selected=[planes[first]]
        extra=(weights*valid*~keep[:,first])@keep
        second=int(np.argmax(extra))
        if second!=first and extra[second]>max(1e-8,.01*score[first]):selected.append(planes[second])
        tangent=_flat_stock_face_tangent(sweep,cloud.positions[i],cloud.normals[i],cloud.tool_ids[i])
        if tangent is not None and not any(np.array_equal(tangent,plane) for plane in selected):
            selected.append(tangent)
        cutter=[];prefix=[]
        for plane in selected:
            cutter.append(prefix+[plane])
            complement=-plane.copy();complement[3]-=1e-7
            prefix=prefix+[complement]
        domains=[]
        for region in base:
            for halfspaces in cutter:
                merged=_reduce_world_region(np.concatenate((region,np.asarray(halfspaces))))
                if merged is not None:domains.append(merged)
        cloud.sweep_domains[i]=tuple(domains)
        changed.append(i)
    return np.asarray(changed,dtype=np.int64)


def _rebuild_child_sweep_domains(cloud,material,boundary_hints=None):
    """New children query only sweeps overlapping their own display footprints.

    Parent-derived planes are never copied. Related past sweeps are reevaluated
    from cutter geometry, in history order, and independent of parent identity.
    """
    started=time.perf_counter();count=len(cloud.positions)
    cloud.inherited_clip=np.zeros((count,2,2,4),dtype=np.float64)
    cloud.inherited_clip[:,0,:,3]=1.;cloud.inherited_clip[:,1,:,3]=-1.
    cloud.sweep_domains=np.empty(count,dtype=object);cloud.sweep_domains[:]=None
    groups={};pairs=0;queries=0
    if count:
        u,v=tangent_frames(cloud.normals)
        radius=3.*cloud.support*(1.+.2*np.clip(cloud.seam_weight,0.,1.))
        extent=radius[:,None]*np.sqrt(u*u+v*v)+1e-7
        import os
        accelerated=os.environ.get('CUTSIM_INITIAL_PREP_ACCEL','1')!='0'
        needs_query=getattr(material,'_strip_work',None) is None
        if accelerated:
            # Cells far from every sweep need no per-GS history query. The
            # same conservative index is used for this enclosing box.
            if needs_query:
                queries+=1
                needs_query=bool(material.query_sweep_ids(
                    np.min(cloud.positions-extent,axis=0),np.max(cloud.positions+extent,axis=0),original_boxes=False))
            planes=[];hits=[]
            for axis in range(3):
                tangent_span=extent[:,axis]-1e-7
                for sign,bound in ((1.,material.stock_lo[axis]),(-1.,material.stock_hi[axis])):
                    plane=np.zeros(4);plane[axis]=sign;plane[3]=-sign*bound
                    planes.append(plane)
                    hits.append(sign*(cloud.positions[:,axis]-bound)<tangent_span-1e-7)
            planes=np.asarray(planes);hits=np.column_stack(hits)
            for i in np.flatnonzero(np.any(hits,axis=1)):
                cloud.sweep_domains[i]=(planes[hits[i]].copy(),)
        for i,p in enumerate(cloud.positions):
            # Initial workpiece faces are also recomputed from the child's own
            # footprint; copying an old parent edge is neither needed nor allowed.
            if not accelerated:
                stock=[]
                for axis in range(3):
                    tangent_span=extent[i,axis]-1e-7
                    for sign,bound in ((1.,material.stock_lo[axis]),(-1.,material.stock_hi[axis])):
                        if sign*(p[axis]-bound)<tangent_span-1e-7:
                            plane=np.zeros(4);plane[axis]=sign;plane[3]=-sign*bound
                            stock.append(plane)
                if stock:cloud.sweep_domains[i]=(np.asarray(stock),)
            if needs_query:
                queries+=1
                for sid in material.query_sweep_ids(p-extent[i],p+extent[i],original_boxes=False):
                    groups.setdefault(sid,[]).append(i)
        if getattr(material,'_strip_work',None) is not None:
            groups=material._strip_work.domain_groups(material,cloud.positions,extent)
        for sid in sorted(groups):
            ids=np.asarray(groups[sid],dtype=np.int64);sweep=material.sweeps[sid]
            ids=ids[np.asarray(sweep.sdf(cloud.positions[ids]))<=radius[ids]+1e-7]
            pairs+=len(ids)
            hints=None if boundary_hints is None else boundary_hints.get(sid)
            _assign_sweep_domains(cloud,sweep,ids,material=material,boundary_planes=hints)
    return dict(index_queries=queries,sweep_pairs=pairs,
                milliseconds=(time.perf_counter()-started)*1000.)


def _sweep_line_interval(planes, index):
    """Feasible interval on one boundary, in normalized Gaussian UV coordinates."""
    q=np.asarray(planes,float).reshape(-1,3);p=q[index]
    foot=-p[2]*p[:2];t=np.array([-p[1],p[0]])
    lo,hi=-np.inf,np.inf
    for j,b in enumerate(q):
        if j==index:continue
        slope=float(b[:2]@t);offset=float(b[:2]@foot+b[2])
        if abs(slope)<1e-12:
            if offset < -1e-9:return None
        elif slope>0:lo=max(lo,-offset/slope)
        else:hi=min(hi,-offset/slope)
    return None if lo>hi+1e-9 else (lo,hi)


def _sweep_local_feasible(planes):
    """Check the full intersection before dropping any boundary (no sampling)."""
    q=np.asarray(planes,float).reshape(-1,3)
    if not len(q) or np.all(q[:,2]>=-1e-9):return True
    return any(_sweep_line_interval(q,i) is not None for i in range(len(q)))


@lru_cache(maxsize=1)
def _sweep_decision_quadrature():
    """Immutable, density-independent 2-sigma samples shared by all children."""
    radius=2.*np.sqrt((np.arange(16)+.5)/16.)
    theta=2.*np.pi*(np.arange(32)+.5)/32.
    points=(radius[:,None,None]*np.stack((np.cos(theta),np.sin(theta)),axis=-1)[None,:,:]).reshape(-1,2)
    weights=np.exp(-.5*np.sum(points*points,axis=1))
    points.setflags(write=False);weights.setflags(write=False)
    return points,weights


def _sweep_top_two(planes, mandatory=None):
    """Boundary peak density first; density times incremental cut effect second.

    These are analytic Sweep/stock constraints, with no counterpart Gaussian.
    Scores therefore use this Gaussian alone, not fabricated peer densities.
    Incremental effects are sampled only inside the 2-sigma decision disk.
    """
    q=np.asarray(planes,float).reshape(-1,3)
    if len(q)<=2:return q.copy(),list(range(len(q)))
    logs=np.full(len(q),-np.inf)
    for j,p in enumerate(q):
        interval=_sweep_line_interval(q,j)
        if interval is not None:
            t=np.clip(0.,*interval)
            logs[j]=-.5*(p[2]**2+t*t)
    canonical=sorted(range(len(q)),key=lambda j:tuple(q[j]))
    first=int(mandatory) if mandatory is not None else max(canonical,key=lambda j:logs[j])
    # Equal-area polar samples; reused for each rare over-limit region.
    points,weights=_sweep_decision_quadrature()
    cut=points@q[:,:2].T+q[:,2]<0.
    marginal=np.sum(weights[:,None]*(cut&~cut[:,first,None]),axis=0)/weights.sum()
    scores=logs+np.log(np.maximum(marginal,1e-12))
    second=max((j for j in canonical if j!=first),key=lambda j:scores[j])
    return q[[first,second]].copy(),[first,second]


def _sweep_draw_regions(regions, original_regions=None):
    """Approximate each convex branch, preserving the two-branch separation."""
    selected=[];indices=[]
    if len(regions)>2:return None,None
    for q in regions:
        keep,idx=_sweep_top_two(q);selected.append(keep);indices.append(idx)
    if len(selected)==2:
        overlap=np.concatenate(selected).copy();overlap[:,2]-=1e-7
        if _sweep_local_feasible(overlap):
            # A dropped separating line must be retained to avoid double alpha.
            original=regions if original_regions is None else original_regions
            pairs=[(i,j) for i,a in enumerate(original[0]) for j,b in enumerate(original[1])
                   if np.linalg.norm(a[:2]+b[:2])<1e-8 and a[2]+b[2]<=1e-9]
            if not pairs:return None,None
            i,j=min(pairs,key=lambda ij:(abs(original[0][ij[0]][2])+abs(original[1][ij[1]][2]),
                                       tuple(original[0][ij[0]]),tuple(original[1][ij[1]])))
            # A separator may have been redundant before approximation. Restore
            # it from the original domain: dropping other edges can make it vital.
            for branch,index in ((0,i),(1,j)):
                q=regions[branch]
                separator=original[branch][index]
                match=next((k for k,p in enumerate(q) if np.max(np.abs(p-separator))<1e-9),None)
                if match is None:q=np.vstack((q,separator));match=len(q)-1
                selected[branch],indices[branch]=_sweep_top_two(q,mandatory=match)
    return selected,indices


def _describe_sweep_domain(domain, position, scale, u, v, exact, *, _projected=None):
    """Pure domain projection; callers own their output/diagnostic containers."""
    regions=[];original_regions=[]
    import os
    accelerated=os.environ.get('CUTSIM_PREP_ACCEL','1')!='0'
    if accelerated and _projected is None:
        from cutsim.rendering.domain_projection import describe_domain
        return describe_domain(domain,position,scale,u,v,exact)
    for region_index,world in enumerate(domain):
        if _projected is not None:
            reduced=_projected[region_index]
            if reduced is None:continue
        else:
            planes=np.column_stack((world[:,:3]@u*scale[0],world[:,:3]@v*scale[1],world[:,:3]@position+world[:,3]))
            reduced=[];empty=False
            for plane in planes:
                length=np.linalg.norm(plane[:2])
                if length<1e-10:
                    if plane[2]<-1e-8:empty=True;break
                    continue
                plane=plane/length
                match=next((j for j,q in enumerate(reduced) if np.linalg.norm(q[:2]-plane[:2])<1e-7),None)
                if match is None:reduced.append(plane)
                elif plane[2]<reduced[match][2]:reduced[match]=plane
            if empty:continue
        if exact:
            # Preserve common constraints until Boolean expansion. Removing
            # individually inactive planes first destroys the shared terms
            # needed to prove that a partition boundary is artificial.
            regions.append(np.asarray(reduced).reshape(-1,3))
        else:
            if not _sweep_local_feasible(reduced):continue
            original_regions.append(np.asarray(reduced).reshape(-1,3))
            regions.append(np.asarray(_sweep_reduce_local(reduced)).reshape(-1,3))
    if exact:
        regions=_sweep_expand_union_regions(regions)
        regions=[q for region in regions if (q:=_sweep_reduce_in_draw_disk(region)) is not None]
        regions=_sweep_expand_union_regions(regions)
        original_regions=regions
    if exact and (len(regions)>2 or any(len(q)>2 for q in regions)):
        selected=None;indices=None
    else:selected,indices=_sweep_draw_regions(regions,original_regions)
    return regions, original_regions, selected, indices


def _apply_sweep_domains(arr,cloud):
    """Keep complete CPU domains; upload important Top-2 for nonempty branches."""
    from cutsim.core.regions import ExactSurfaceDomains
    ids=[i for i,d in enumerate(cloud.sweep_domains) if d is not None]
    if not ids:return
    import os
    accelerated=os.environ.get('CUTSIM_PREP_ACCEL','1')!='0'
    u,v=_render_axes(arr['rot'])
    from cutsim.rendering import domain_batch
    descriptions=domain_batch.describe_many(cloud,arr,u,v,ids) if domain_batch.enabled() else {}
    for i in ids:
        exact=isinstance(cloud.sweep_domains[i],ExactSurfaceDomains)
        from cutsim.rendering.exact_draw_cache import domain_cache, array_key
        domain=cloud.sweep_domains[i]
        if i in descriptions:
            description=descriptions[i]
        elif accelerated:
            description=_describe_sweep_domain(domain,arr['positions'][i],arr['scale'][i],u[i],v[i],exact)
        else:
            key=(exact, tuple(array_key(q) for q in domain), array_key(arr['positions'][i]),
                 array_key(arr['scale'][i]), array_key(u[i]), array_key(v[i]))
            description=domain_cache.get(key)
            if description is None:
                description=_describe_sweep_domain(domain,arr['positions'][i],arr['scale'][i],u[i],v[i],exact)
                domain_cache.put(key,description)
        regions,original_regions,selected,indices=description
        source=getattr(domain,'source_planes',None)
        if selected is not None and source is not None:
            from cutsim.rendering.stock_clipping import complete_clip_bounds
            local=np.column_stack((source[:,:3]@u[i]*arr['scale'][i,0],
                source[:,:3]@v[i]*arr['scale'][i,1],source[:,:3]@arr['positions'][i]+source[:,3]))
            lengths=np.linalg.norm(local[:,:2],axis=1)
            local=local[lengths>1e-10]/lengths[lengths>1e-10,None]
            selected=[complete_clip_bounds(planes,local) for planes in selected]
        pending=selected is None
        arr['clip_empty'][i]=not regions
        arr['clip_pending'][i]=pending;arr['clip_reason'][i]=5 if pending else 0
        arr['clip_concave'][i]=not pending and len(selected)==2
        arr['clip_sweep_top2'][i]=not pending and any(len(q)>2 for q in regions)
        if pending or arr['clip_sweep_top2'][i]:
            arr.approximated.setdefault(i,{}).update(sweep_regions_world=cloud.sweep_domains[i],
                sweep_selected_indices=indices,sweep_selected_planes=selected)
        if pending:
            arr.approximated[i]['fallback_regions_local']=tuple(original_regions)
            if exact:arr.approximated[i]['finite_strip_domain']=True
            # Both the stock box and the generating face's axial limits are
            # mandatory, including at the smallest draw-only subdivision.
            bounds=[getattr(domain,name,None) for name in ('stock_planes','source_planes')]
            bounds=[planes for planes in bounds if planes is not None]
            stock=np.vstack(bounds) if bounds else None
            if stock is not None:
                local=np.column_stack((stock[:,:3]@u[i]*arr['scale'][i,0],
                    stock[:,:3]@v[i]*arr['scale'][i,1],stock[:,:3]@arr['positions'][i]+stock[:,3]))
                lengths=np.linalg.norm(local[:,:2],axis=1);local=local[lengths>1e-10]/lengths[lengths>1e-10,None]
                arr.approximated[i]['stock_planes_local']=local
        if pending or not regions:continue
        arr['clip_approximated'][i]=arr['clip_sweep_top2'][i]
        values=[local_planes_to_clip_values(q) for q in selected]
        for key,value in zip(CLIP_ATTRIBUTE_NAMES,values[0]):arr[key][i,0]=value
        arr['clip_second_values'][i]=values[1] if len(values)==2 else [4.,0.,4.,0.]
        arr['clip_edge_count'][i]=max(len(q) for q in selected)


def _sweep_expand_union_regions(regions):
    """Remove Boolean partition planes without changing the region union.

    R=(A and p), S=(B and not p), B subset A implies R or S == A or S.
    Repeatedly apply this identity, then remove absorbed conjunctions. Exact
    normalized plane tuples identify complements: no angular binning, shifted
    boundary, area threshold or sampling establishes the equivalence. This
    recovers overlapping analytic branches from a disjoint convex difference.
    CPU domains remain untouched. The input's finite-support reduction is the
    only reason equivalence is restricted to the drawing footprint.
    """
    terms=[set(map(tuple,np.asarray(region).reshape(-1,3))) for region in regions]
    if not terms:return []
    if any(not term for term in terms):return [np.empty((0,3))]
    owners={}
    for i,term in enumerate(terms):
        for plane in term:owners.setdefault(plane,set()).add(i)
    changed=True
    while changed:
        changed=False
        for i,term in enumerate(terms):
            for plane in sorted(term):
                opposite=tuple(-x for x in plane)
                rest=term-{plane}
                for j in tuple(owners.get(opposite,())):
                    if i!=j and (terms[j]-{opposite})<=rest:
                        term.remove(plane);owners[plane].discard(i)
                        changed=True;break
            if not term:return [np.empty((0,3))]
    kept=[]
    for i,term in enumerate(terms):
        if any(other<=term and (other!=term or j<i) for j,other in enumerate(terms) if j!=i):continue
        kept.append(np.asarray(sorted(term),float).reshape(-1,3))
    return kept


def _sweep_reduce_in_draw_disk(planes):
    """Keep every active halfplane in a conservative 3-sigma bounding square.

    A discarded plane is strictly inactive on the full feasible polygon, so
    removing it changes no draw fragment. The Float32 collar is render-only.
    Touching planes are retained, including constraints on very thin regions.
    """
    from cutsim.rendering.clip_fallback import _clip_polygon
    q=np.asarray(planes,float).reshape(-1,3)
    q=q[q[:,2]<3.+1e-6]
    radius=3.+1e-6
    polygon=np.array([[-radius,-radius],[radius,-radius],[radius,radius],[-radius,radius]])
    for plane in q:
        polygon=_clip_polygon(polygon,plane)
        if not len(polygon):return None
    if not len(q):return q
    minimum=np.min(polygon@q[:,:2].T+q[:,2],axis=0)
    return q[minimum<=1e-7]


def _sweep_reduce_local(planes):
    """Remove only globally redundant linear constraints; no finite-radius shortcut."""
    planes=[p.copy() for p in planes]
    def vertices(q):
        # Identical all-pairs linear feasibility, batched in NumPy. Parent
        # footprint constraints can have 32 edges; a Python loop per vertex
        # made their exact global reduction dominate drawing preparation.
        a,b=np.triu_indices(len(q),1);a,b=q[a],q[b]
        det=a[:,0]*b[:,1]-a[:,1]*b[:,0]
        good=np.abs(det)>=1e-10;a,b,det=a[good],b[good],det[good]
        points=np.column_stack(((a[:,1]*b[:,2]-a[:,2]*b[:,1])/det,
                                (a[:,2]*b[:,0]-a[:,0]*b[:,2])/det))
        return points[np.all(points@q[:,:2].T+q[:,2]>=-1e-9,axis=1)]
    while len(planes)>2:
        removed=False
        for j,objective in enumerate(planes):
            other=np.asarray(planes[:j]+planes[j+1:]);points=vertices(other)
            if not len(points):continue
            # A linear objective has a minimum at a vertex unless it decreases
            # along a feasible recession ray. Check every candidate ray.
            t=np.column_stack((-other[:,1],other[:,0]))
            directions=np.concatenate((t,-t,-objective[None,:2]))
            directions/=np.maximum(np.linalg.norm(directions,axis=1)[:,None],1e-30)
            feasible=np.all(other[:,:2]@directions.T>=-1e-10,axis=0)
            if np.any((directions@objective[:2]<-1e-10)&feasible):continue
            if np.min(points@objective[:2]+objective[2])>=1e-8:
                planes.pop(j);removed=True;break
        if not removed:break
    return planes


def _cut_parent_visible(probes,parent_rows,attributes):
    """Evaluate pre-cut parent's uploaded AND/OR region in its original UV frame."""
    from cutsim.cutting.footprint import _CUT_FOOTPRINT_SETTINGS
    ids=np.asarray(parent_rows,dtype=np.int64)
    u,v=_render_axes(attributes['rot'][ids])
    delta=probes-attributes['positions'][ids,None,:]
    uv=np.stack((np.einsum('nkj,nj->nk',delta,u)/attributes['scale'][ids,0,None],
                 np.einsum('nkj,nj->nk',delta,v)/attributes['scale'][ids,1,None]),axis=-1)
    def region(values):
        keep=np.ones(uv.shape[:2],bool)
        for k in (0,2):
            angle=values[:,k]; split=values[:,k+1]
            distance=uv[:,:,0]*np.cos(angle[:,None])+uv[:,:,1]*np.sin(angle[:,None])-split[:,None]
            keep &= (angle[:,None]==4.) | (distance>=-1e-7)
        return keep
    first=np.column_stack([attributes[name][ids,0] for name in CLIP_ATTRIBUTE_NAMES])
    keep=region(first)
    if np.any(attributes['clip_concave'][ids]):
        keep |= attributes['clip_concave'][ids,None]&region(attributes['clip_second_values'][ids])
    keep &= ~(attributes['clip_pending'][ids,None]|attributes.get('clip_empty',np.zeros(len(attributes['positions']),bool))[ids,None])
    return keep&(np.sum(uv*uv,axis=-1)<=_CUT_FOOTPRINT_SETTINGS.intersection_sigma**2+1e-6)


def _clip_preservable_parents(before,rows,sweep,parent_clip,material):
    """Prove that a partial parent can use the existing renderer without splitting.

    Keep only nonempty, exact (no Top-2 omission) Sweep domains, with at most two
    disjoint draw branches and two planes per branch. The same assignment runs
    on actual retained parents later; this temporary cloud does not mutate them.
    """
    if not len(rows):return np.zeros(0,bool)
    cloud=_local_take(before,rows)
    regions=_clip_world_regions(parent_clip)
    changed=_assign_sweep_domains(cloud,sweep,np.arange(len(rows)),parent_regions=regions,material=material)
    trial=NativeClipArrays({k:v.copy() for k,v in parent_clip.items()})
    trial.approximated={};trial.pending={}
    _apply_sweep_domains(trial,cloud)
    keep=np.zeros(len(rows),bool);keep[changed]=True
    keep &= ~(trial['clip_pending']|trial['clip_empty']|trial['clip_sweep_top2'])
    # Do not rescue a hidden parent by treating its absent original region as full.
    keep &= ~(parent_clip['clip_pending']|parent_clip.get('clip_empty',False))
    return keep


def _edge_group_same_boundary(planes,eligible,log_scores,tangent_scales,radius_sigma,angle_deg,concave):
    """Group compatible nested kept regions; use the least restrictive line.

    Containment is analytic over the normalized evaluation disk, not a center
    distance or sample test. Priority between groups is their maximum original
    density/sharpness score; group size and boundary length never add weight.
    Complete-link compatibility prevents a chain of small turns merging a corner.
    """
    planes=np.asarray(planes,float);scales=np.asarray(tangent_scales,float)
    if not np.isfinite(angle_deg) or not 0.<=angle_deg<90.:
        raise ValueError('Similar-edge angle must be in [0,90) degrees')
    if not np.isfinite(radius_sigma) or radius_sigma<=0.:
        raise ValueError('Similar-edge evaluation radius must be positive')
    eligible=np.asarray(eligible,bool);log_scores=np.asarray(log_scores,float)
    count=len(planes);ids=np.flatnonzero(eligible)
    representative=np.full(count,-1,np.int32);representative[ids]=ids
    ranked=np.where(eligible,log_scores,-np.inf)
    if angle_deg==0. or len(ids)<2:return representative,ranked
    physical=planes[:,:2]/scales
    physical/=np.maximum(np.linalg.norm(physical,axis=1)[:,None],1e-30)
    similar=physical@physical.T>=math.cos(math.radians(angle_deg))-1e-12
    unit=planes/np.maximum(np.linalg.norm(planes[:,:2],axis=1)[:,None],1e-30)
    a,b,c=unit.T;radius=float(radius_sigma)
    determinant=a[:,None]*b[None,:]-a[None,:]*b[:,None]
    nonparallel=np.abs(determinant)>1e-10
    x=np.divide(b[:,None]*c[None,:]-b[None,:]*c[:,None],determinant,out=np.zeros_like(determinant),where=nonparallel)
    y=np.divide(c[:,None]*a[None,:]-c[None,:]*a[:,None],determinant,out=np.zeros_like(determinant),where=nonparallel)
    local_corner=nonparallel&(x*x+y*y<=radius*radius+1e-10)
    # Any OR participant keeps its own identity; swapping its line could change
    # its join with a third boundary even when this pair appears compatible.
    protected=np.any(concave,axis=1)
    compatible=similar&~local_corner&~protected[:,None]&~protected[None,:]
    # min(n_j.X+c_j) over disk intersect kept half-plane i. If the unconstrained
    # disk minimum is outside i, the minimum is at an endpoint of i's chord.
    dot=unit[:,:2]@unit[:,:2].T
    chord=np.sqrt(np.maximum(0.,radius*radius-c*c))
    minimum=c[None,:]-c[:,None]*dot-chord[:,None]*np.abs(determinant)
    unrestricted=(c[:,None]-radius*dot)>=0.
    minimum=np.where(unrestricted,c[None,:]-radius,minimum)
    # Do not merge tangent/empty/full-disk constraints; real positive-length
    # finite candidates cross the disk, but generic diagnostics may not.
    crossing=np.abs(c)<radius-1e-10
    contains=(minimum>=0.)&crossing[:,None]&crossing[None,:] # [i,j]: kept_i subset kept_j
    # A containing circular cap has at least as large a normalized signed
    # offset. Unlike abs(c), this remains correct when the center is cut away.
    order=sorted(ids,key=lambda i:(-float(c[i]),float(a[i]),float(b[i])))
    groups=[]
    for i in order:
        target=next((g for g in groups if contains[i,g[0]] and all(compatible[i,j] for j in g)),None)
        if target is None:groups.append([int(i)])
        else:target.append(int(i))
    ranked[:]=-np.inf
    for group in groups:
        rep=group[0]
        representative[group]=rep
        ranked[rep]=float(np.max(log_scores[group]))
    return representative,ranked


def _edge_density_peak(ends,own_position,own_axes,own_scale,own_opacity,
                       peer_positions,peer_axes,peer_scales,peer_opacity,*,radius_sigma=None):
    """Exact maximum of two unclipped Gaussians on each finite world segment.

    X(t)=E0+t*(E1-E0), 0<=t<=1. Concatenated normalized tangent
    coordinates z(t)=z0+t*dz give t*=clip(-dot(z0,dz)/dot(dz,dz)).
    With radius_sigma, restrict t to BOTH Gaussians' normalized tangent disks.
    A maximum outside the disks moves to the feasible interval's endpoint;
    an empty common interval scores zero (-inf in log space).
    No line-length multiplier, sampling, screen filtering or alpha compositing.
    A zero-length segment is evaluated as a point by this numerical helper;
    the geometric candidate builder excludes such segments before scoring.
    """
    ends=np.asarray(ends,float).reshape(-1,2,3)
    own_axes=np.asarray(own_axes,float).reshape(2,3)
    peer_axes=np.asarray(peer_axes,float).reshape(-1,2,3)
    own_scale=np.asarray(own_scale,float).reshape(2)
    peer_scales=np.asarray(peer_scales,float).reshape(-1,2)
    peer_positions=np.asarray(peer_positions,float).reshape(-1,3)
    if len(peer_axes)!=len(ends) or len(peer_scales)!=len(ends) or len(peer_positions)!=len(ends):
        raise ValueError('One peer Gaussian per finite segment is required')
    values=(ends,own_position,own_axes,own_scale,own_opacity,
            peer_positions,peer_axes,peer_scales,peer_opacity)
    if any(not np.isfinite(v).all() for v in values) or np.any(own_scale<=0) or np.any(peer_scales<=0):
        raise ValueError('Gaussian data must be finite with positive tangent scales')
    delta=ends[:,1]-ends[:,0]
    a=(ends[:,0]-own_position)@own_axes.T/own_scale
    da=delta@own_axes.T/own_scale
    b=np.einsum('ni,nji->nj',ends[:,0]-peer_positions,peer_axes)/peer_scales
    db=np.einsum('ni,nji->nj',delta,peer_axes)/peer_scales
    numerator=np.sum(a*da+b*db,axis=1)
    denominator=np.sum(da*da+db*db,axis=1)
    t=np.clip(np.divide(-numerator,denominator,out=np.zeros(len(ends)),where=denominator>0),0.,1.)
    own_sq=np.sum((a+t[:,None]*da)**2,axis=1)
    peer_sq=np.sum((b+t[:,None]*db)**2,axis=1)
    valid=np.ones(len(ends),bool)
    if radius_sigma is not None:
        radius=float(radius_sigma)
        if not np.isfinite(radius) or radius<=0:raise ValueError('Density radius must be positive and finite')
        # Existing finite world segments usually already satisfy the limit.
        # Solve disk/line intervals only for out-of-range unconstrained maxima.
        bad=np.flatnonzero((own_sq>radius*radius)|(peer_sq>radius*radius))
        if len(bad):
            low=np.zeros(len(bad));high=np.ones(len(bad))
            for z,dz in ((a[bad],da[bad]),(b[bad],db[bad])):
                speed=np.sum(dz*dz,axis=1);moving=speed>0
                center=np.divide(-np.sum(z*dz,axis=1),speed,out=np.zeros(len(bad)),where=moving)
                closest=z+center[:,None]*dz
                room=radius*radius-np.sum(closest*closest,axis=1)
                half=np.sqrt(np.divide(np.maximum(room,0.),speed,out=np.zeros(len(bad)),where=moving))
                low=np.maximum(low,np.where(moving,center-half,-np.inf))
                high=np.minimum(high,np.where(moving,center+half,np.inf))
                valid[bad]&=room>=0.
            valid[bad]&=high>=low
            t[bad]=np.where(valid[bad],np.minimum(np.maximum(t[bad],low),high),0.)
            own_sq[bad]=np.sum((a[bad]+t[bad,None]*da[bad])**2,axis=1)
            peer_sq[bad]=np.sum((b[bad]+t[bad,None]*db[bad])**2,axis=1)
    log_own=-np.logaddexp(0.,-np.asarray(own_opacity,float))-.5*own_sq
    log_peer=-np.logaddexp(0.,-np.asarray(peer_opacity,float))-.5*peer_sq
    return dict(log_peak=np.where(valid,log_own+log_peer,-np.inf),position=ends[:,0]+t[:,None]*delta,
                own_alpha=np.where(valid,np.exp(log_own),0.),peer_alpha=np.where(valid,np.exp(log_peer),0.),
                fraction=t,valid=valid,own_radius_sigma=np.sqrt(own_sq),peer_radius_sigma=np.sqrt(peer_sq))


def _edge_peer_frames(normals,support,seam_weight,settings):
    """Match the current unsplit native pack's Float32 rotation and scales.

    Native parents have circular tangent scales and one shared settings.alpha.
    Explicit axes/scales are nevertheless used by the peak solver, which also
    supports anisotropic Gaussians. No extra shading or full-cloud pack is done.
    """
    u,v=tangent_frames(normals)
    rot=rotation_matrices_to_quaternions(np.stack([u,v,normals],axis=2)).astype(np.float32)
    u,v=_render_axes(rot)
    sigma=np.asarray(support)*settings.sigma_scale
    sigma*=1.+np.clip(seam_weight,0.,1.)*(max(float(settings.resolution_seam_scale),1.)-1.)
    sigma=sigma.astype(np.float32).astype(float)
    return np.stack([u,v],axis=1),np.column_stack([sigma,sigma])


def _edge_merge_density(entry,peaks,j,peer_position):
    """A merged line takes its strongest peer, never sums peer count/length."""
    log_peak=float(peaks['log_peak'][j])
    # Geometry-only tie breaking: stable under neighbour/store traversal changes.
    key=(-log_peak,*map(float,peer_position),*map(float,peaks['position'][j]))
    if 'density_key' not in entry or key<entry['density_key']:
        entry.update(density_key=key,log_density_peak=log_peak,
                     density_peak_position=peaks['position'][j].copy(),
                     density_peer_position=np.asarray(peer_position).copy(),
                     density_own_alpha=float(peaks['own_alpha'][j]),
                     density_peer_alpha=float(peaks['peer_alpha'][j]),
                     density_valid=bool(peaks['valid'][j]),
                     density_own_radius_sigma=float(peaks['own_radius_sigma'][j]),
                     density_peer_radius_sigma=float(peaks['peer_radius_sigma'][j]))


def _ridge_signed_mask(da,db,r,s,band_fraction=0.,slack=0.):
    """Keep strict ridges; tolerate a near-centre plane residual only one-sided.

    Small mixed signs are sampling-uncertain, not proven ridges. Accept them only
    when the other signed distance is clearly negative. Positive/positive valleys
    and two indeterminate on-plane centres remain excluded. No cutter data used.
    """
    if not np.isfinite(band_fraction) or not 0.<=band_fraction<=.02:
        raise ValueError('Edge centre band fraction must be in [0,0.02]')
    tol=1e-7*np.maximum(np.maximum(r,s),1e-5)
    strict=(da < -tol+slack)&(db < -tol+slack)
    if band_fraction==0.:return strict
    band=band_fraction*np.minimum(r,s)
    boundary=(da<=band+slack)&(db<=band+slack)&((da < -band)|(db < -band))
    return strict|boundary


def _edge_signed_mask(da,db,r,s,band_fraction=0.,slack=0.,*,cosine=None,valley_min_angle_deg=5.,clip_valleys=True):
    """Admit both ridge and valley signs; reject coincident/ambiguous pairs."""
    ridge=_ridge_signed_mask(da,db,r,s,band_fraction,slack)
    if not clip_valleys:
        return ridge
    angle=float(valley_min_angle_deg)
    if not np.isfinite(angle) or not 0.<=angle<=180.:
        raise ValueError('Valley normal-angle threshold must be in [0,180] degrees')
    valley=_ridge_signed_mask(-da,-db,r,s,band_fraction,slack)
    if cosine is not None:
        valley &= np.asarray(cosine)<=math.cos(math.radians(angle))+1e-12
    return ridge|valley

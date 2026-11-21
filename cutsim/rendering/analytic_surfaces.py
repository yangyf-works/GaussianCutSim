"""Immutable finite generating surfaces, independent of continuity surfaceID.

Equation: (x-origin)' Q (x-origin) + linear.(x-origin) + constant = 0.
Cut surfaces use the outward CUTTER gradient; stock uses outward STOCK.
Only geometry referenced by live GS survives in the current snapshot.
"""
from dataclasses import dataclass, replace
import numpy as np
from cutsim.rendering.profile_cut_clip import pack_cutters


@dataclass(frozen=True, eq=False)
class AnalyticSolid:
    packed: tuple
    block: int
    source: str = 'sweep'


@dataclass(frozen=True, eq=False)
class AnalyticPatch:
    equation: np.ndarray  # origin(3), Q(9), linear(3), constant(1)
    solid: AnalyticSolid
    part: int
    face: int            # -1 quadric, otherwise packed plane index
    stock: bool = False


def freeze(a):
    a=np.ascontiguousarray(a);a.flags.writeable=False;return a


def patch_ids(s):
    ids=getattr(s,'analytic_patch_ids',None)
    return np.full(len(s.endpoints['positions']),-1,np.int64) if ids is None else ids


def set_metadata(s,ids,table):
    s.analytic_patch_ids=freeze(np.asarray(ids,np.int64))
    s.analytic_patches={int(i):table[int(i)] for i in np.unique(ids) if i>=0}


def select_metadata(source,result,keep):
    if getattr(source,'analytic_patch_ids',None) is not None:
        set_metadata(result,source.analytic_patch_ids[keep],source.analytic_patches)


def combine_metadata(old,keep,new,result):
    if getattr(old,'analytic_patch_ids',None) is None and getattr(new,'analytic_patch_ids',None) is None:return
    table=dict(getattr(old,'analytic_patches',None) or {})
    for key,value in (getattr(new,'analytic_patches',None) or {}).items():
        if key in table and table[key] is not value:
            # The same block can attach base/adaptive/gap data separately.
            if not np.array_equal(table[key].equation,value.equation):raise ValueError('Analytic patch ID collision')
        table[key]=value
    set_metadata(result,np.r_[patch_ids(old)[keep],patch_ids(new)],table)


def initialize_stock(snapshot,low,high):
    """Attach the six finite box planes; their negative side is stock interior."""
    if getattr(snapshot,'analytic_patch_ids',None) is not None:return
    planes=[]
    for axis in range(3):
        for sign,bound in ((1.,low[axis]),(-1.,high[axis])):
            p=np.zeros(5);p[axis]=sign;p[3]=-sign*bound;p[4]=1.;planes.append(p)
    packed=(np.zeros((1,14)),np.array([[0,6]],np.int64),np.array(planes),np.array([np.r_[low,high]]),np.zeros((1,13)))
    solid=AnalyticSolid(tuple(freeze(a) for a in packed),-1,source='stock');table={}
    for i,h in enumerate(planes):
        eq=np.zeros(16);eq[12:15]=-h[:3];eq[15]=-h[3]
        table[i+1]=AnalyticPatch(freeze(eq),solid,0,i,True)
    ids=snapshot.gaussian.attributes['surface_id'][:,0].astype(np.int64)
    set_metadata(snapshot,ids,table)


def sweep_patches(sweep,block,packed=None):
    packed=pack_cutters((sweep,)) if packed is None else packed
    solid=AnalyticSolid(tuple(freeze(a) for a in packed),block)
    records,spans,planes=packed[:3];table={};slot=0
    for part,(rec,span) in enumerate(zip(records,spans)):
        faces=list(range(*span)) + ([-1] if rec[13] else [])
        for face in faces:
            eq=np.zeros(16)
            if face<0:eq[:3]=rec[:3];eq[3:12]=rec[3:12];eq[15]=-rec[12]**2
            else:eq[12:15]=-planes[face,:3];eq[15]=-planes[face,3]
            if slot>=1024:raise ValueError('Too many analytic faces in a single sweep')
            table[100+1_000_000*block+slot]=AnalyticPatch(freeze(eq),solid,part,face);slot+=1
    return table


def generating_pose(sweep,p,n):
    """Recover a genuine generating tool pose on a sloped move, if certified.

    Existing ramp Dexels can sample an intermediate tool pose rather than the
    final envelope. Keep that actual source equation, never fit a free plane
    to an unmatched point. No GS geometry or cutting decision changes here.
    """
    indexed = sweep if hasattr(sweep, 'local') else None
    world_p, world_n = p, n
    if indexed is not None:
        p = (p-indexed.origin)@indexed.basis
        n = n@indexed.basis
        sweep = indexed.local
    if not hasattr(sweep,'a_xyz'):return None
    a=np.asarray(sweep.a_xyz);direction=np.asarray(sweep.b_xyz)-a
    if sweep.kind=='ball' and n[2]>1e-6:
        tip=p+sweep.radius*n-np.array([0.,0.,sweep.radius])
        t=float((tip-a)@direction)/max(float(direction@direction),1e-30)
    elif abs(n[2])>1.-1e-6 and abs(direction[2])>1e-12:
        height=sweep.cutting_start if n[2]>0 else sweep.cutting_end
        t=float((p[2]-height-a[2])/direction[2]);tip=a+t*direction
    elif abs(n[2])<1e-6 and direction[:2]@direction[:2]>1e-20:
        xy=p[:2]+sweep.radius*n[:2];t=float((xy-a[:2])@direction[:2])/(direction[:2]@direction[:2])
        tip=a+t*direction
        if np.linalg.norm(tip[:2]-xy)>1e-6:return None
    else:return None
    if not -1e-8<=t<=1+1e-8 or np.linalg.norm(tip-(a+t*direction))>1e-6:return None
    tip=a+np.clip(t,0.,1.)*direction
    stationary=replace(sweep,a_xyz=tip,b_xyz=tip)
    if indexed is not None:
        stationary = replace(indexed, local=stationary)
        p, n = world_p, world_n
    candidates=sweep_patches(stationary,0);best=None;error=np.inf
    for patch in candidates.values():
        e=patch.equation;d=p-e[:3];q=e[3:12].reshape(3,3);g=2*d@q+e[12:15];length=np.linalg.norm(g)
        if length<1e-10 or -g@n/length<1-1e-6:continue
        value=abs(d@q@d+d@e[12:15]+e[15])/length
        rec,span=patch.solid.packed[0][patch.part],patch.solid.packed[1][patch.part]
        h=patch.solid.packed[2][span[0]:span[1]]
        if np.any(p@h[:,:3].T+h[:,3]<-1e-6):continue
        if rec[13] and (p-rec[:3])@rec[3:12].reshape(3,3)@(p-rec[:3])>rec[12]**2+1e-6:continue
        if value<min(error,1e-6):best=patch;error=value
    return best


def assign_generated(snapshot,sweep,block,*,packed=None):
    """Identify only newly generated GS; children inherit without re-fitting."""
    ids=patch_ids(snapshot).copy();rows=np.flatnonzero(ids<0)
    if not len(rows):return
    table=dict(getattr(snapshot,'analytic_patches',None) or {});new=sweep_patches(sweep,block,packed)
    p=snapshot.endpoints['positions'][rows];n=snapshot.endpoints['normals'][rows]
    sigma=snapshot.gaussian.attributes['scale'][rows,0].astype(float)
    best=np.full(len(rows),np.inf);chosen=np.full(len(rows),-1,np.int64)
    # Small face count, vectorized assignment. Intersection/edge work is CUDA.
    for key,patch in new.items():
        eq=patch.equation;d=p-eq[:3];q=eq[3:12].reshape(3,3)
        gradient=2*d@q+eq[12:15];length=np.linalg.norm(gradient,axis=1)
        value=np.einsum('ij,ij->i',d@q,d)+d@eq[12:15]+eq[15]
        aligned=-np.sum(gradient*n,axis=1)/np.maximum(length,1e-30)
        rec,span=patch.solid.packed[0][patch.part],patch.solid.packed[1][patch.part]
        planes=patch.solid.packed[2][span[0]:span[1]]
        field=p@planes[:,:3].T+planes[:,3]
        violation=np.maximum(0.,np.max(-field/np.maximum(np.linalg.norm(planes[:,:3],axis=1),1e-30),axis=1,initial=0.))
        if rec[13]:
            delta=p-rec[:3];radial=np.sqrt(np.maximum(0.,np.einsum('ij,ij->i',delta@rec[3:12].reshape(3,3),delta)))
            violation=np.maximum(violation,radial-rec[12])
        # Existing supplementary GS may be slightly off the finite envelope
        # near chart joins. Associate the nearest compatible physical face;
        # do not change the GS position or invent an unbounded tangent plane.
        score=np.abs(value)/np.maximum(length,1e-30)+violation+sigma*np.maximum(0.,1.-aligned)
        take=(aligned>.95)&(score<best)&(score<np.maximum(1e-4,.5*sigma))
        best[take]=score[take];chosen[take]=key
    for row in np.flatnonzero(chosen<0):
        patch=generating_pose(sweep,p[row],n[row])
        if patch is None:continue
        # Share identical pose faces; reserve a separate range in this block.
        patch=replace(patch,solid=replace(patch.solid,block=block,source='tool_pose'))
        key=next((k for k,v in {**table,**new}.items() if v.solid.block==block and
            v.solid.source=='tool_pose' and np.array_equal(v.equation,patch.equation) and
            np.array_equal(v.solid.packed[2],patch.solid.packed[2])),None)
        if key is None:
            base=100+1_000_000*block
            key=max([base+1023]+[k for k in (*table,*new) if base+1024<=k<base+1_000_000])+1
            if key>=base+5120:raise ValueError('Generating-pose analytic patch budget exceeded')
            new[key]=patch
        chosen[row]=key
    if np.any(chosen<0):
        bad=np.flatnonzero(chosen<0)[:3]
        raise ValueError(f'{np.count_nonzero(chosen<0)} generated GS have no analytic generating face: positions={p[bad].tolist()}, normals={n[bad].tolist()}')
    ids[rows]=chosen;table.update(new);set_metadata(snapshot,ids,table)


def pack_snapshot(snapshot, *, rows=None):
    """Dense GPU surface table and finite solid domains, uploaded once per pass."""
    ids=patch_ids(snapshot)
    selected=ids if rows is None else ids[np.asarray(rows,np.int64)]
    if np.any(selected<0):raise ValueError('Missing analytic surface metadata; reset and recompute')
    keys=np.unique(selected);table=snapshot.analytic_patches
    dense=np.full(len(ids),-1,np.int64)
    if not len(keys):
        return (dense,np.empty((0,16)),np.empty((0,4),np.int64),
                np.empty((0,14)),np.empty((0,2),np.int64),np.empty((0,5)))
    locations=np.searchsorted(keys,ids)
    match=(locations<len(keys))&(keys[np.minimum(locations,len(keys)-1)]==ids)
    dense[match]=locations[match]
    equations=[];meta=[];records=[];spans=[];planes=[];solid_map={};record_count=plane_count=0
    for key in keys:
        patch=table[int(key)];solid=patch.solid;stamp=id(solid)
        if stamp not in solid_map:
            r,s,h=solid.packed[:3];solid_map[stamp]=(record_count,len(r))
            records.append(r);spans.append(s+plane_count);planes.append(h)
            record_count+=len(r);plane_count+=len(h)
        start,count=solid_map[stamp]
        equations.append(patch.equation);meta.append([start,count,start+patch.part,int(patch.stock)])
    return (dense,np.array(equations),np.array(meta,np.int64),
        np.concatenate(records),np.concatenate(spans),np.concatenate(planes))

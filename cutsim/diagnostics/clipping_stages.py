"""Optional immutable CPU clipping inputs and GS generation provenance."""
from copy import copy
import numpy as np
from cutsim.rendering.clip_search import CLIP_SIGMA
from cutsim.rendering.profile_cut_clip import ProfileClip
from cutsim.cutting.tool_profile_state import RAY_STRIDE

SOURCE_NAMES=('Stock','Base Dexel','Adaptive / seam','Residual repair','Gap fill','Clipping subdivision')
SOURCE_COLORS=np.array([[.6,.6,.6],[.3,.65,1],[1,.5,.1],[.8,.4,1],[.2,.9,.55],[1,.85,.15]],np.float32)


def snapshot_copy(snapshot):
    result=copy(snapshot);result.endpoints=dict(snapshot.endpoints)
    for name in ('clipping','material_clipping'):
        clip=getattr(snapshot,name,None)
        if clip is not None:setattr(result,name,ProfileClip(clip.planes,dict(clip.stats)))
    return result


def before_clipping(snapshot,previous,kept):
    result=snapshot_copy(snapshot);nold=int(kept.sum());n=len(snapshot.endpoints['positions'])
    for name in ('clipping','material_clipping'):
        planes=np.tile([0.,0,np.inf,0,0,np.inf,np.inf],(n,1))
        old=getattr(previous,name,None) or previous.clipping
        planes[:nold]=old.planes[kept];planes.flags.writeable=False
        setattr(result,name,ProfileClip(planes,{}))
    return result


def remap_stages(debug,block,mapping):
    for name in ('before_clip','before_split'):
        snapshot=debug.get(name)
        if snapshot is None:continue
        ids=snapshot.endpoints['ray_ids'].copy();local=ids-(block+1)*RAY_STRIDE
        rows=(snapshot.endpoints['owners']==block)&(local>=0)&(local<len(mapping))
        if np.any(mapping[local[rows]]<0):raise AssertionError('Adaptive parent in clipping debug snapshot')
        ids[rows]=mapping[local[rows]]+(block+1)*RAY_STRIDE;ids.flags.writeable=False
        snapshot.endpoints['ray_ids']=ids


def source_codes(snapshot,deltas):
    ids=snapshot.endpoints['ray_ids'];owner=snapshot.endpoints['owners']
    code=np.ones(len(ids),np.uint8);code[owner<0]=0
    local=-ids-(owner+1)*RAY_STRIDE;synthetic=(ids<0)&(owner>=0)
    code[synthetic]=3;code[synthetic&(local>=100_000)]=4;code[synthetic&(local>=200_000)]=5
    for block in np.unique(owner[(owner>=0)&(ids>=0)]):
        if block>=len(deltas):continue
        delta=deltas[int(block)];meta=getattr(delta,'adaptive',None)
        if meta is not None:
            rows=(owner==block)&(ids>=(block+1)*RAY_STRIDE+meta['base_rays'])
            code[rows]=2
    return code


def gs_description(snapshot,row,deltas):
    ids=snapshot.endpoints;owner=int(ids['owners'][row]);ray=int(ids['ray_ids'][row])
    sigma=float(snapshot.gaussian.attributes['scale'][row,0]);position=ids['positions'][row]
    code=int(source_codes(snapshot,deltas)[row])
    origin=f'Born in block: {owner+1}' if owner>=0 else 'Born on initial stock'
    lines=[f'Source: {SOURCE_NAMES[code]}',f'{origin} | GS row: {row}',
           'XYZ: '+', '.join(f'{x:.5f}' for x in position)+' mm',
           f'Sigma: {sigma:.6g} mm | Clip domain diameter ({2*CLIP_SIGMA:g} sigma): {2*CLIP_SIGMA*sigma:.6g} mm']
    if ray>=0 and owner>=0:lines.append(f'Local ray: {ray-(owner+1)*RAY_STRIDE}')
    elif owner>=0:lines.append('Generated support GS; no Dexel ray.')
    if code==5 and 0<=owner<len(deltas):
        debug=getattr(deltas[owner],'clipping_debug',None);child=-ray-(owner+1)*RAY_STRIDE-200_000
        if debug is not None and 0<=child<len(debug.get('split_parents',())):
            parent=int(debug['split_parents'][child]);source=debug['before_split']
            parent_size=float(source.gaussian.attributes['scale'][parent,0])
            lines.append(f'Parent before subdivision: row {parent}, sigma {parent_size:.6g} mm')
            lines.append(f'Child / parent size: {sigma/parent_size:.3f}')
    if snapshot.clipping is not None:
        planes=snapshot.clipping.planes[row]
        lines.append('Clip distances: '+', '.join(f'{x:.6g}' for x in planes[[2,5]])+' mm')
    if getattr(snapshot,'analytic_patch_ids',None) is not None:
        patch_id=int(snapshot.analytic_patch_ids[row]);patch=snapshot.analytic_patches.get(patch_id)
        lines.append(f'SurfaceID: {snapshot.material_surface_ids[row]} | Analytic patch ID: {patch_id}')
        if patch is not None:
            kind='Plane' if not np.any(patch.equation[3:12]) else 'Quadric'
            if patch.stock:
                axis=int(np.argmax(abs(patch.equation[12:15])))
                normal=patch.equation[12+axis]
                bound=patch.equation[axis]-patch.equation[15]/normal
                lines.append(f'Initial stock plane: {"XYZ"[axis]} {"max" if normal>0 else "min"} | {"xyz"[axis]} = {bound:.6g} mm')
                lines.append('Finite stock plane participates in analytic clipping.')
            else:
                lines.append(f'{kind} ({patch.solid.source}) | Original generating block: {patch.solid.block+1} | finite part {patch.part}')
    return '\n'.join(lines)

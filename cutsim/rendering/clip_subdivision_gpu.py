"""GPU edge packing, normalization, classification and bounded subdivision.

Uses the SAME double-precision certificates, witness samples and greedy cover
as clip_subdivision.partition. CPU handles batch descriptors and publication;
no Python loop normalizes individual GS. Payloads stay on CUDA until publication.
"""
from time import perf_counter
import numpy as np
from cutsim.rendering.clip_search import CLIP_SIGMA
from cutsim.rendering.profile_cut_clip import ProfileClip
from cutsim.rendering.clip_edge_batches import batches
from cutsim.rendering.clip_subdivision import CUDA, ClippingSubdivisionError


PREPARE=r'''
extern "C" __global__ void pack_edges(const long long* rows,const double* input,
 int count,int lines,int width,double* raw,int* lengths){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 long long row=rows[i];int offset=lengths[row];lengths[row]+=lines;
 for(int j=0;j<3*lines;j++)raw[(row*width+offset)*3+j]=input[i*lines*3+j];
}
extern "C" __global__ void normalize_edges(const double* raw,const int* lengths,
 const double* radius,int count,int width,double* normalized,int* kept){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 const double* source=raw+3*width*i;double* dest=normalized+3*width*i;int n=0;
 for(int j=0;j<lengths[i];j++){
  double x=source[3*j],y=source[3*j+1],d=source[3*j+2];
  // Match numpy.linalg.norm on two components, including operation order.
  double length=sqrt(x*x+y*y);
  if(length<1e-12&&d<0.){dest[0]=dest[1]=0.;dest[2]=-1.;for(int k=1;k<width;k++){dest[3*k]=dest[3*k+1]=0.;dest[3*k+2]=INF;}kept[i]=1;return;}
  if(length<=1e-12||!isfinite(d))continue;
  x/=length;y/=length;d/=length;
  if(d<radius[i]+1e-9){dest[3*n]=x;dest[3*n+1]=y;dest[3*n+2]=d;n++;}
 }
 kept[i]=n;
}
'''


def partition_gpu(snapshot,*,max_children=4096,per_parent=32):
    from cutsim.cuda.profile_dexel_cuda import runtime
    from cutsim.rendering.gs_area_retirement import _disc
    started=perf_counter();rt=runtime();cp=rt.cp
    radius=CLIP_SIGMA*snapshot.gaussian.attributes['scale'][:,0].astype(float);n=len(radius)
    rawd=batches(snapshot.clipping.constraints);rawm=batches(snapshot.material_clipping.constraints)
    rows=np.union1d(rawd.keys(),rawm.keys())
    # A missing override means the existing two-edge certificate, not no limit.
    rawd=batches_copy(rawd);rawm=batches_copy(rawm)
    for raw,clip in ((rawd,snapshot.clipping),(rawm,snapshot.material_clipping)):
        missing=np.setdiff1d(rows,raw.keys())
        raw.append(missing,clip.planes[missing,:6].reshape(-1,2,3))
    descriptors=[];max_width=1
    for raw in (rawd,rawm):
        count=np.zeros(len(rows),np.int32);blocks=[]
        for ids,edges in raw.blocks:
            local=np.searchsorted(rows,ids);np.add.at(count,local,edges.shape[1])
            blocks.append((local,edges))
        max_width=max(max_width,int(count.max(initial=0)));descriptors.append(blocks)
    # Explicit workspace bound; never truncate constraints to fit memory.
    workspace_bytes=4*len(rows)*max_width*3*8
    if workspace_bytes>192*1024*1024:
        raise ClippingSubdivisionError('GPU edge preparation exceeds 192 MiB; block not published')
    parents=[];children=[];errors=[];split_control_reads=0
    with rt.context():
        if not hasattr(rt,'clip_batch_kernels'):
            module=cp.RawModule(code=CUDA+PREPARE,options=('--fmad=false',))
            rt.clip_batch_kernels=[module.get_function(x) for x in
                ('pack_edges','normalize_edges','roots','probes','candidates','select_cover')]
        pack,normalize,root,probe,candidate,select=rt.clip_batch_kernels
        rr_all=cp.asarray(radius[rows]);edges=[];lengths=[]
        for blocks in descriptors:
            raw=cp.empty((len(rows),max_width,3),cp.float64);length=cp.zeros(len(rows),cp.int32)
            for ids,values in blocks:
                pack(((len(ids)+127)//128,),(128,),
                    (cp.asarray(ids),cp.asarray(values,order='C'),np.int32(len(ids)),np.int32(values.shape[1]),np.int32(max_width),raw,length))
            output=cp.zeros_like(raw);output[:,:,2]=cp.inf;kept=cp.empty(len(rows),cp.int32)
            if len(rows):normalize(((len(rows)+127)//128,),(128,),
                (raw,length,rr_all,np.int32(len(rows)),np.int32(max_width),output,kept))
            edges.append(output);lengths.append(kept)
        flags=cp.empty(len(rows),cp.uint8);certificates=cp.empty((len(rows),15),cp.float64)
        group_first=cp.empty(len(rows),cp.int64)
        sizes=cp.maximum(cp.maximum(lengths[0],lengths[1]),1)
        too_many=cp.count_nonzero(sizes>256)
        # The reference visits groups in first-occurrence order. Remember that
        # order on GPU and restore it before assigning published child IDs.
        split_groups=[]
        for width in (1,2,4,8,16,32,64,128,256):
            ids=cp.flatnonzero((sizes<=width)&(sizes>(width//2)))
            if not len(ids):continue
            group_first[ids]=ids[0]
            d=padded(edges[0],ids,width,cp);m=padded(edges[1],ids,width,cp);rr=rr_all[ids]
            f=cp.empty(len(ids),cp.uint8);c=cp.empty((len(ids),15),cp.float64)
            root(((len(ids)+127)//128,),(128,),(d,m,rr,np.int32(len(ids)),np.int32(width),f,c))
            flags[ids]=f;certificates[ids]=c
            split=cp.flatnonzero(f==2)
            if len(split):split_groups.append((width,ids[split],d[split],m[split],rr[split]))
        angles=(np.arange(128)+.5)*np.pi/64
        samples=cp.asarray(np.vstack((_disc(16,32),np.column_stack((np.cos(angles),np.sin(angles))))))
        for width,all_ids,all_d,all_m,all_r in split_groups:
            for start in range(0,len(all_ids),128):
                ids=all_ids[start:start+128];d=all_d[start:start+128];m=all_m[start:start+128];rr=all_r[start:start+128];count=len(ids)
                ns=len(samples);total=ns+2*(2*width+width*(width-1)//2)
                xy=cp.empty((count,total,2),cp.float64);valid=cp.zeros((count,total),cp.uint8)
                probe(((count*total+127)//128,),(128,),
                    (d,m,rr,np.int32(count),np.int32(width),samples,np.int32(ns),xy,valid))
                length=cp.sum(valid,axis=1,dtype=cp.int32)
                maxlen=int(cp.asnumpy(length.max(),stream=rt.stream));split_control_reads+=1
                if not maxlen:raise ClippingSubdivisionError('Unresolved clipping domain has no witness samples')
                packed=cp.zeros((count,maxlen,2),cp.float64);a,b=cp.nonzero(valid)
                offsets=cp.cumsum(valid,axis=1,dtype=cp.int32)-1;packed[a,offsets[a,b]]=xy[a,b]
                cand=cp.empty((count,2*maxlen,15),cp.float64);cv=cp.empty((count,2*maxlen),cp.uint8)
                candidate(((count*2*maxlen+127)//128,),(128,),
                    (d,m,rr,np.int32(count),np.int32(width),packed,length,np.int32(maxlen),cand,cv))
                result=cp.empty((count,per_parent,15),cp.float64);counts=cp.empty(count,cp.int32);miss=cp.empty(count,cp.int32)
                select((count,),(128,),(packed,length,np.int32(maxlen),cand,cv,np.int32(per_parent),result,counts,miss),shared_mem=maxlen)
                a,b=cp.nonzero(cp.arange(per_parent)[None,:]<counts[:,None])
                parents.append(ids[a]);children.append(result[a,b]);errors.append(cp.sum(miss))
        misses=cp.sum(cp.stack(errors)) if errors else cp.asarray(0)
        control=cp.asnumpy(cp.stack((too_many,misses)),stream=rt.stream)
        if control[0]:raise ClippingSubdivisionError('More than 256 active clipping edges; block not published')
        if control[1]:raise ClippingSubdivisionError(f'Clipping needs more subdivision: {control[1]} uncovered witnesses; block not published')
        if sum(len(x) for x in children)>max_children:
            raise ClippingSubdivisionError(f'Clipping subdivision budget {max_children} exceeded; block not published')
        flags=cp.asnumpy(flags,stream=rt.stream);c=cp.asnumpy(certificates,stream=rt.stream)
        if parents:
            pp=cp.concatenate(parents);cc=cp.concatenate(children)
            order=cp.lexsort(cp.stack((cp.arange(len(pp)),pp,group_first[pp])))
            pp=cp.asnumpy(pp[order],stream=rt.stream);cc=cp.asnumpy(cc[order],stream=rt.stream)
        else:pp=np.empty(0,np.int64);cc=np.empty((0,15))
    keep=np.ones(n,bool);keep[rows[flags!=1]]=False
    draw=snapshot.clipping.planes.copy();material=snapshot.material_clipping.planes.copy();good=flags==1
    for target,sl in ((draw,slice(3,9)),(material,slice(9,15))):
        target[rows[good],:6]=c[good,sl];target[rows[good],6]=radius[rows[good]]
    empty=~np.isfinite(material[:,2])&(material[:,2]<0);material[empty]=[0,0,-1,0,0,np.inf,0]
    stats=dict(snapshot.clipping.stats,clip_subdivision=True,clip_gpu_prepare=True,
        clip_partition_checked=len(rows),clip_partition_kept=int(good.sum()),clip_partition_parents=int((flags==2).sum()),
        clip_partition_children=len(cc),clip_partition_ms=(perf_counter()-started)*1000,radius_limited_gs=0,
        clip_prepare_workspace_bytes=workspace_bytes,clip_prepare_batches=sum(map(len,descriptors)),
        clip_partition_result_readbacks=4 if parents else 2,clip_partition_publication_phases=1,
        clip_partition_split_control_reads=split_control_reads)
    snapshot.clipping=ProfileClip(draw,stats);snapshot.material_clipping=ProfileClip(material,dict(stats))
    return keep,rows[pp],cc,stats


def batches_copy(raw):
    from cutsim.rendering.clip_edge_batches import EdgeBatches
    return EdgeBatches(raw.blocks)


def padded(values,ids,width,cp):
    # The normalized count can be smaller than the raw storage width. Padding
    # is necessary for power-of-two groups wider than that raw storage width.
    if width<=values.shape[1]:return cp.ascontiguousarray(values[ids,:width])
    out=cp.zeros((len(ids),width,3),cp.float64);out[:,:,2]=cp.inf
    out[:,:values.shape[1]]=values[ids];return out

"""Conservative cutting-boundary clips for profile Dexel surfels.

Each swept cutter is a union of convex plane/PSD-quadric primitives. For each
primitive choose one separating halfplane on the GS tangent plane. Quadrics
use a tangent at the radially projected boundary, including when the GS center
has been cut but part of its disc survives. Intersecting these
exterior halfplanes cannot draw inside any cutter, up to upload roundoff.

The native renderer accepts two lines. Retain those nearest the center and
limit the 2-sigma clipping domain by every other line. This is conservative: it can
remove additional surface coverage near curved seams, not reconstruct it.
No sampled material intervals or endpoint positions are modified. Explicit
cutter-exempt rows bypass this cutter-separation guarantee; stock limits still
apply. The current-GS caller uses that policy for matching physical surface IDs.
"""
from dataclasses import dataclass
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from cutsim.rendering.clip_search import CLIP_SIGMA

from cutsim.cutting.local_dexel import bounds
from cutsim.geometry.strip_intersections import swept_primitives


@dataclass(frozen=True)
class ProfileClip:
    # Each row is [nx1,ny1,d1,nx2,ny2,d2,remaining_disk_radius].
    # n.local_xy + d >= 0; unused distances/radius are +inf.
    planes: np.ndarray
    stats: dict
    # Transient uncompressed halfplanes, indexed by row; never cutter history.
    constraints: object = None


def _minimum_on_sweep(sweep, normal):
    """Exact support value; distinguish physical caps from partition planes.

    A plane coincident with a GS can only certify an exterior if the WHOLE
    cutter is on its other side. At the ball equator, for example, the sphere
    and shaft both have boundary planes but their union has interior there.
    """
    from cutsim.core.cutting_sweep import CuttingSweep
    from cutsim.core.sweep import BallSweep,FlatSweep
    if hasattr(sweep, 'local'):
        return float(-sweep.support(-np.asarray(normal)))
    h=float(np.linalg.norm(normal[:2]));z=float(normal[2]);r=sweep.radius
    if isinstance(sweep,CuttingSweep):
        values=[];lo=sweep.cutting_start;hi=sweep.cutting_end
        if sweep.kind=='ball' and lo<min(hi,r):
            height=np.clip(r-r*z/max(np.linalg.norm(normal),1e-30),lo,min(hi,r))
            values.append(z*height-h*np.sqrt(max(0.,r*r-(height-r)**2)))
        bottom=max(lo,r) if sweep.kind=='ball' else lo
        if hi>=bottom:values.append(min(z*bottom,z*hi)-h*r)
        return min(normal@sweep.a_xyz,normal@sweep.b_xyz)+min(values)
    xy=min(normal[:2]@sweep.a_xy,normal[:2]@sweep.b_xy)
    if type(sweep) is FlatSweep:
        return xy+min(z*sweep.z_lo,z*sweep.z_hi)-h*r
    if type(sweep) is BallSweep:
        value=z*sweep.center_z-r*np.linalg.norm(normal)
        if sweep.z_hi>=sweep.center_z:
            value=min(value,min(z*sweep.center_z,z*sweep.z_hi)-h*r)
        return xy+value
    raise TypeError(f'Unsupported cutting-boundary support: {type(sweep)}')


def _support_record(sweep):
    from cutsim.core.cutting_sweep import CuttingSweep
    from cutsim.core.sweep import FlatSweep,BallSweep
    if isinstance(sweep,CuttingSweep) or hasattr(sweep, 'local'):
        axis = sweep.basis[:,2] if hasattr(sweep, 'local') else [0.,0.,1.]
        return np.r_[sweep.a_xyz,sweep.b_xyz,sweep.radius,sweep.cutting_start,sweep.cutting_end,float(sweep.kind=='ball'),axis]
    if type(sweep) is FlatSweep:return np.r_[sweep.a_xy,0.,sweep.b_xy,0.,sweep.radius,sweep.z_lo,sweep.z_hi,0.,0.,0.,1.]
    if type(sweep) is BallSweep:return np.r_[sweep.a_xy,sweep.center_z,sweep.b_xy,sweep.center_z,sweep.radius,0.,sweep.z_hi-sweep.center_z,2.,0.,0.,1.]
    raise TypeError(type(sweep))


def _support_minimum(s,n):
    z=n@s[10:13];h=np.linalg.norm(n-z[:,None]*s[10:13],axis=1);r,lo,hi,kind=s[6:10]
    base=np.minimum(n@s[:3],n@s[3:6])
    if kind==2:
        value=-r*np.linalg.norm(n,axis=1)
        if hi>=0:value=np.minimum(value,np.minimum(0,z*hi)-h*r)
        return base+value
    value=np.full(len(n),np.inf)
    if kind==1 and lo<min(hi,r):
        height=np.clip(r-r*z/np.maximum(np.linalg.norm(n,axis=1),1e-30),lo,min(hi,r))
        value=z*height-h*np.sqrt(np.maximum(0,r*r-(height-r)**2))
    bottom=max(lo,r) if kind==1 else lo
    if hi>=bottom:value=np.minimum(value,np.minimum(z*bottom,z*hi)-h*r)
    return base+value


def pack_cutters(sweeps):
    records, planes, spans, boxes, supports = [], [], [], [], []
    offset = 0
    seen = set()
    for sweep in sweeps:
        lo, hi = bounds(sweep)
        for part in swept_primitives(sweep):
            record = np.zeros(14)
            record[:3] = part.center
            if part.matrix is not None:
                # The separating-plane proof requires a convex quadric.
                if np.linalg.eigvalsh(part.matrix).min() < -1e-10:
                    raise ValueError('Cut clipping requires positive semidefinite quadrics')
                record[3:12] = part.matrix.ravel()
                record[13] = 1
            record[12] = part.radius
            q = np.asarray(part.planes, dtype=np.float64).reshape(-1, 4)
            external=[_minimum_on_sweep(sweep,plane[:3])+plane[3]>=-1e-9*max(1.,np.linalg.norm(plane[:3])) for plane in q]
            q=np.column_stack((q,external))
            key = (record.tobytes(), q.tobytes())
            if key in seen:
                continue
            seen.add(key)
            records.append(record); planes.append(q); boxes.append(np.r_[lo, hi]);supports.append(_support_record(sweep))
            spans.append((offset, offset+len(q))); offset += len(q)
    return (np.asarray(records, float).reshape(-1, 14),
            np.asarray(spans, np.int64).reshape(-1, 2),
            np.concatenate(planes) if planes else np.empty((0, 5)),
            np.asarray(boxes, float).reshape(-1, 6),np.asarray(supports,float).reshape(-1,13))


def _insert(out, x, y, d):
    """Vectorized CPU reference / cheap display-only section insertion."""
    x, y, d = (np.broadcast_to(a, (len(out),)).copy() for a in (x, y, d))
    for slot in (0, 3):
        same = (x*out[:, slot]+y*out[:, slot+1] > 1.-1e-12)
        redundant = same & (d >= out[:, slot+2])
        d[redundant] = np.inf
        replace = same & ~redundant
        out[replace, slot+2] = d[replace]
        d[replace] = np.inf
    swap = out[:, 5] < out[:, 2]
    out[swap, :6] = out[swap, :6][:, [3,4,5,0,1,2]]
    for slot in (0, 3):
        take = d < out[:, slot+2]
        old = out[take, slot:slot+3].copy()
        out[take, slot:slot+3] = np.column_stack((x[take], y[take], d[take]))
        x[take], y[take], d[take] = old.T
    out[:, 6] = np.minimum(out[:, 6], d)


def _cpu_planes(p, u, v, max_radius, packed, low, high, cutter_exempt=None, raw=None):
    out = np.zeros((len(p), 7)); out[:, [2,5,6]] = np.inf
    normals=np.cross(u,v)
    for axis in range(3):
        for sign, bound in ((1., low[axis]), (-1., high[axis])):
            x, y = sign*u[:, axis], sign*v[:, axis]
            norm = np.hypot(x, y)
            signed = sign*(p[:, axis]-bound)
            # A tangent plane parallel to and outside the stock has no support.
            # Cutter identity exemptions must never exempt stock constraints.
            # At the stock boundary the material side must lie inside stock.
            # An inward-facing tool cap would put its material side outside.
            outside=(signed < -1e-9)|((abs(signed)<=1e-9)&(sign*normals[:,axis]>0))
            out[(norm<=1e-10)&outside, 6] = 0.
            d = np.divide(signed, norm, out=np.full(len(p), np.inf), where=norm>1e-10)
            if raw is not None:
                raw[:,2*axis+int(sign<0),:]=np.column_stack((x/np.maximum(norm,1e-30),y/np.maximum(norm,1e-30),d))
                raw[(norm<=1e-10)&outside,2*axis+int(sign<0)]=[0,0,-1]
            _insert(out, x/np.maximum(norm, 1e-30), y/np.maximum(norm, 1e-30), d)
    records, spans, planes, boxes, supports = packed
    for part,(record, span, box,support) in enumerate(zip(records, spans, boxes,supports)):
        near = np.all((p+max_radius[:, None] >= box[:3]) & (p-max_radius[:, None] <= box[3:]), axis=1)
        if cutter_exempt is not None:near &= ~cutter_exempt
        ids = np.flatnonzero(near)
        if not len(ids):
            continue
        n=normals[ids];projection=np.sum(p[ids]*n,axis=1)
        # A plane supporting the WHOLE convex sweep cannot intersect its
        # interior. Do not clip it at artificial cap/barrel partition planes.
        exterior=(_support_minimum(support,n)-projection>=-1e-9)|(_support_minimum(support,-n)+projection>=-1e-9)
        ids=ids[~exterior]
        if not len(ids):continue
        pp, uu, vv = p[ids], u[ids], v[ids]
        best = np.full(len(ids), -np.inf); bx = np.zeros(len(ids)); by = bx.copy()
        # Geometry is evaluated at the Float64 material endpoints. Small
        # signed roundoff at a tangency is treated as zero, not as penetration.
        for plane in planes[span[0]:span[1]]:
            norm3 = np.linalg.norm(plane[:3])
            if norm3==0:
                if plane[3]<0:best[:]=np.inf
                continue
            value = -(pp@plane[:3]+plane[3]); x = -uu@plane[:3]; y = -vv@plane[:3]
            norm = np.hypot(x, y)
            eps=1e-9*max(norm3,1e-30)
            is_parallel=norm<=1e-10*max(norm3,1e-30)
            parallel=is_parallel & ((value>eps) | ((plane[4]!=0)&(value>=-eps)))
            good = ~is_parallel
            value=np.where(abs(value)<=eps,0.,value)
            d = np.divide(value, norm, out=np.full(len(ids), -np.inf), where=good)
            d[parallel] = np.inf
            take = d > best; best[take] = d[take]
            bx[take] = x[take]/np.maximum(norm[take], 1e-30); by[take] = y[take]/np.maximum(norm[take], 1e-30)
        if record[13]:
            delta = pp-record[:3]; g = delta@record[3:12].reshape(3,3)
            q2 = np.maximum(0.,np.sum(delta*g, axis=1))
            # Q is PSD: q.Q.x >= r*sqrt(q.Q.q) certifies the exterior.
            # Unlike linearizing f at p, this plane touches the actual quadric.
            value = q2-record[12]*np.sqrt(q2)
            x = np.sum(g*uu, axis=1); y = np.sum(g*vv, axis=1); norm = np.hypot(x,y)
            tolerance = 1e-9*max(record[12]**2, 1.)
            parallel = (norm <= 1e-9*max(record[12], 1.)) & (q2-record[12]**2 >= -tolerance)
            good = norm > 1e-9*max(record[12],1.)
            value=np.where(abs(q2-record[12]**2)<=tolerance,0.,value)
            d = np.divide(value,norm,out=np.full(len(ids),-np.inf),where=good)
            d[parallel] = np.inf
            take = d > best; best[take] = d[take]
            bx[take] = x[take]/np.maximum(norm[take],1e-30); by[take] = y[take]/np.maximum(norm[take],1e-30)
        # Negative d clips away the center, preserving a certified off-center
        # fragment. No certificate means that this disc cannot be published.
        local = out[ids].copy()
        local[np.isneginf(best), 6] = 0.
        empty=np.isneginf(best)
        best[(best > max_radius[ids]) | empty] = np.inf
        if raw is not None:
            raw[ids,6+part,:]=np.column_stack((bx,by,best))
            raw[ids[empty],6+part,:]=[0,0,-1]
        _insert(local, bx, by, best); out[ids] = local
    return out


_KERNEL = r'''
#define INFINITY (__longlong_as_double(0x7ff0000000000000LL))
__device__ void insert_clip(double* a, double x, double y, double d) {
    for(int k=0;k<6;k+=3) {
        if(x*a[k]+y*a[k+1]>1.-1e-12) {
            a[k+2]=fmin(a[k+2],d);
            if(a[5]<a[2]) for(int j=0;j<3;j++){double t=a[j];a[j]=a[j+3];a[j+3]=t;}
            return;
        }
    }
    for(int k=0;k<6;k+=3) if(d<a[k+2]) {
        double xx=a[k], yy=a[k+1], dd=a[k+2];
        a[k]=x; a[k+1]=y; a[k+2]=d; x=xx; y=yy; d=dd;
    }
    a[6]=fmin(a[6],d);
}
__device__ double dot_clip(const double* a,const double* b) {return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
__device__ double support_clip(const double* s,const double* n) {
    double z=dot_clip(n,s+10),r=s[6],lo=s[7],hi=s[8],kind=s[9];
    double radial[3];for(int k=0;k<3;k++)radial[k]=n[k]-z*s[10+k];
    double h=sqrt(dot_clip(radial,radial));
    double norm=sqrt(dot_clip(n,n)),base=fmin(dot_clip(s,n),dot_clip(s+3,n)),value=INFINITY;
    if(kind==2.) {
        value=-r*norm;if(hi>=0.)value=fmin(value,fmin(0.,z*hi)-h*r);return base+value;
    }
    if(kind==1. && lo<fmin(hi,r)) {
        double height=fmin(fmin(hi,r),fmax(lo,r-r*z/fmax(norm,1e-30)));
        value=z*height-h*sqrt(fmax(0.,r*r-(height-r)*(height-r)));
    }
    double bottom=kind==1.?fmax(lo,r):lo;
    if(hi>=bottom)value=fmin(value,fmin(z*bottom,z*hi)-h*r);
    return base+value;
}
extern "C" __global__ void profile_clip(const double* positions,const double* us,const double* vs,
    const double* radius,long long count,const double* records,const long long* spans,
    const double* planes,const double* boxes,const double* supports,int parts,const double* stock,
    const unsigned char* cutter_exempt,double* result) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=count)return;
    const double *p=positions+3*i,*u=us+3*i,*v=vs+3*i;
    double normal[3]={u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]};
    double negative[3]={-normal[0],-normal[1],-normal[2]},projection=dot_clip(p,normal);
    double a[7]={0,0,INFINITY,0,0,INFINITY,INFINITY};
    for(int k=0;k<3;k++) for(int side=0;side<2;side++) {
        double s=side?-1.:1.,x=s*u[k],y=s*v[k],n=hypot(x,y);
        double signed_distance=s*(p[k]-stock[k+side*3]);
        if(n>1e-10) insert_clip(a,x/n,y/n,signed_distance/n);
        else if(signed_distance < -1e-9 || (fabs(signed_distance)<=1e-9 && s*normal[k]>0.)) a[6]=0.;
    }
    for(int j=0;j<parts && !cutter_exempt[i];j++) {
        const double* box=boxes+6*j; bool near=true;
        for(int k=0;k<3;k++) if(p[k]+radius[i]<box[k] || p[k]-radius[i]>box[k+3])near=false;
        if(!near)continue;
        if(support_clip(supports+13*j,normal)-projection>=-1e-9 ||
           support_clip(supports+13*j,negative)+projection>=-1e-9)continue;
        double best=-INFINITY,bx=0,by=0;
        for(long long h=spans[2*j];h<spans[2*j+1];h++) {
            const double* plane=planes+5*h;
            double value=-(dot_clip(p,plane)+plane[3]),x=-dot_clip(u,plane),y=-dot_clip(v,plane);
            double n=hypot(x,y),n3=sqrt(dot_clip(plane,plane));
            if(n3==0.){if(plane[3]<0.){best=INFINITY;break;}continue;}
            {
                if(n<=1e-10*fmax(n3,1e-30)) {
                    if(value>1e-9*fmax(n3,1e-30) || (plane[4] && value>=-1e-9*fmax(n3,1e-30))){best=INFINITY;break;}
                    continue;
                }
                double d=(fabs(value)<=1e-9*fmax(n3,1e-30)?0.:value)/n;
                if(d>best){best=d;bx=x/n;by=y/n;}
            }
        }
        const double* r=records+14*j;
        if(r[13] && best!=INFINITY) {
            double q[3]={p[0]-r[0],p[1]-r[1],p[2]-r[2]},g[3];
            for(int k=0;k<3;k++)g[k]=dot_clip(r+3+3*k,q);
            double q2=fmax(0.,dot_clip(q,g)),value=q2-r[12]*sqrt(q2),x=dot_clip(g,u),y=dot_clip(g,v),n=hypot(x,y);
            double tol=1e-9*fmax(r[12]*r[12],1.);
            if(n<=1e-9*fmax(r[12],1.)) {if(q2-r[12]*r[12]>=-tol)best=INFINITY;}
            else {double d=(fabs(q2-r[12]*r[12])<=tol?0.:value)/n;if(d>best){best=d;bx=x/n;by=y/n;}}
        }
        if(best==-INFINITY)a[6]=0.;
        else if(best<=radius[i])insert_clip(a,bx,by,best);
    }
    double roundoff=8.*1.1920928955078125e-7*(1.+fmax(fabs(p[0]),fmax(fabs(p[1]),fabs(p[2])))+radius[i]);
    a[2]-=roundoff;a[5]-=roundoff;a[6]-=roundoff;
    for(int k=0;k<6;k+=3)if(a[k+2]>radius[i]){a[k]=a[k+1]=0.;a[k+2]=INFINITY;}
    if(a[6]>radius[i])a[6]=INFINITY;
    for(int k=0;k<7;k++)result[7*i+k]=a[k];
}
'''


class CutterClipCache:
    """Append-only CPU packing cache for immutable simulator sweep objects.

    Global deduplication preserves the FIRST primitive's box/support, exactly
    as pack_cutters(full_history). A replaced/truncated prefix starts over.
    No Dexel intervals or GPU allocations are retained here.
    """
    def __init__(self):self.clear()

    def clear(self):
        self.sweeps=();self.records=[];self.spans=[];self.planes=[];self.boxes=[];self.supports=[]
        self.seen=set();self.offset=0;self.packed=None

    def pack(self,sweeps):
        sweeps=tuple(sweeps)
        if len(sweeps)<len(self.sweeps) or any(a is not b for a,b in zip(self.sweeps,sweeps)):self.clear()
        changed=self.packed is None
        for sweep in sweeps[len(self.sweeps):]:
            records,spans,planes,boxes,supports=pack_cutters([sweep])
            for record,span,box,support in zip(records,spans,boxes,supports):
                q=planes[span[0]:span[1]];key=(record.tobytes(),q.tobytes())
                if key in self.seen:continue
                self.seen.add(key);changed=True
                self.records.append(record);self.planes.append(q);self.boxes.append(box);self.supports.append(support)
                self.spans.append((self.offset,self.offset+len(q)));self.offset+=len(q)
        self.sweeps=sweeps
        if changed:
            self.packed=(np.asarray(self.records,float).reshape(-1,14),np.asarray(self.spans,np.int64).reshape(-1,2),
                np.concatenate(self.planes) if self.planes else np.empty((0,5)),
                np.asarray(self.boxes,float).reshape(-1,6),np.asarray(self.supports,float).reshape(-1,13))
            for a in self.packed:a.flags.writeable=False
        return self.packed


def prepare_cut_clipping(snapshot, sweeps, low, high, backend='cpu',*,packed_cutters=None,device_cutters=None,cutter_exempt=None,collect_edges=False):
    """Prepare once; cutter-exempt rows retain stock limits in the same pass."""
    started = perf_counter()
    sweeps=tuple(sweeps)
    p = np.ascontiguousarray(snapshot.endpoints['positions'], dtype=float)
    exempt=np.zeros(len(p),np.uint8) if cutter_exempt is None else np.ascontiguousarray(cutter_exempt,dtype=np.uint8)
    if exempt.shape!=(len(p),):raise ValueError('Cutter exemption must have one flag per GS')
    u, v = (np.ascontiguousarray(a, dtype=float) for a in (snapshot.gaussian.u, snapshot.gaussian.v))
    # The published clip domain is independent of the display-size control.
    max_radius = CLIP_SIGMA*snapshot.gaussian.attributes['scale'][:,0].astype(float)
    packed = pack_cutters(sweeps) if packed_cutters is None else packed_cutters
    pack_ms = (perf_counter()-started)*1000
    kernel_ms = None
    raw=None
    if backend == 'cuda':
        from cutsim.cuda.profile_dexel_cuda import runtime
        rt = runtime(); cp = rt.cp
        with rt.context():
            key='cut_clip_edges_kernel' if collect_edges else 'cut_clip_identity_kernel'
            if not hasattr(rt,key):
                code=_KERNEL
                if collect_edges:
                    code=code.replace('unsigned char* cutter_exempt,double* result)', 'unsigned char* cutter_exempt,double* result,double* raw)')
                    code=code.replace('double a[7]={', 'double* edges=raw+i*(parts+6)*3;\n    double a[7]={')
                    code=code.replace('if(n>1e-10) insert_clip(a,x/n,y/n,signed_distance/n);',
                        'if(n>1e-10){insert_clip(a,x/n,y/n,signed_distance/n);int e=3*(2*k+side);edges[e]=x/n;edges[e+1]=y/n;edges[e+2]=signed_distance/n;}')
                    code=code.replace('else if(best<=radius[i])insert_clip(a,bx,by,best);',
                        'else if(best<=radius[i]){insert_clip(a,bx,by,best);int e=3*(6+j);edges[e]=bx;edges[e+1]=by;edges[e+2]=best;}')
                    code=code.replace(') a[6]=0.;', ') {a[6]=0.;edges[3*(2*k+side)+2]=-1.;}')
                    code=code.replace('if(best==-INFINITY)a[6]=0.;', 'if(best==-INFINITY){a[6]=0.;edges[3*(6+j)+2]=-1.;}')
                    code=code.replace('a[2]-=roundoff;', 'for(int e=0;e<parts+6;e++)edges[3*e+2]-=roundoff;\n    a[2]-=roundoff;')
                setattr(rt,key,cp.RawKernel(code,'profile_clip',options=('--fmad=false',)))
            cutters=tuple(cp.asarray(a) for a in packed) if device_cutters is None else device_cutters
            data = [*(cp.asarray(a) for a in (p,u,v,max_radius)),*cutters,cp.asarray(np.r_[low,high])]
            result = cp.empty((len(p),7),cp.float64)
            extra=()
            if collect_edges:
                edges=cp.zeros((len(p),len(packed[0])+6,3),cp.float64);edges[:,:,2]=cp.inf;extra=(edges,)
            begin, end = cp.cuda.Event(), cp.cuda.Event(); begin.record()
            if len(p):
                getattr(rt,key)(((len(p)+127)//128,), (128,),
                    (*data[:4],np.int64(len(p)),*data[4:9],np.int32(len(packed[0])),data[9],cp.asarray(exempt),result,*extra))
            end.record(); end.synchronize(); kernel_ms = float(cp.cuda.get_elapsed_time(begin,end))
            out = cp.asnumpy(result,stream=rt.stream,blocking=True)
            if collect_edges:raw=cp.asnumpy(edges,stream=rt.stream)
    elif backend == 'cpu':
        if collect_edges:
            raw=np.zeros((len(p),len(packed[0])+6,3));raw[:,:,2]=np.inf
        out = _cpu_planes(p,u,v,max_radius,packed,np.asarray(low),np.asarray(high),exempt.astype(bool),raw)
    else:
        raise ValueError('Clipping backend must be cpu or cuda')
    # Make d conservative for FP32 center/quaternion/clip-line uploads. The
    # allowance is a display tolerance only; material intervals stay Float64.
    if backend=='cpu':
        roundoff = 8*np.finfo(np.float32).eps*(1+np.max(np.abs(p),axis=1)+max_radius)
        out[:, [2,5,6]] -= roundoff[:,None]
        if raw is not None:raw[:,:,2]-=roundoff[:,None]
        # Constraints beyond the maximum display disk are irrelevant.
        for slot in (0,3):
            far=out[:,slot+2]>max_radius
            out[far,slot:slot+2]=0.;out[far,slot+2]=np.inf
        out[out[:,6]>max_radius,6]=np.inf
    out.flags.writeable = False
    elapsed = (perf_counter()-started)*1000
    sigma = snapshot.gaussian.attributes['scale'][:,0]
    stats = dict(backend=backend,primitives=len(packed[0]),pack_ms=pack_ms,kernel_ms=kernel_ms,
                 total_ms=elapsed,clipped_gs=int(np.count_nonzero(out[:,2]<CLIP_SIGMA*sigma)),
                 radius_limited_gs=int(np.count_nonzero(out[:,6]<CLIP_SIGMA*sigma)),
                 history_count=len(sweeps),stock_bounds=tuple(np.r_[low,high].tolist()),
                 input_gs=len(p),recomputed_gs=len(p),reused_gs=0,selection_ms=0.,mode='full')
    return ProfileClip(out,stats,raw)


def update_cut_clipping(snapshot,previous,kept,sweeps,low,high,backend='cpu',*,cutter_cache=None):
    """Recompute changed supports; copy immutable certificates for other GS.

    Caller contract: snapshot starts with previous[kept], with identical world
    positions, tangent frames and base scales, followed by newly added GS.
    Re-evaluate affected rows against ALL history; never accumulate rounded,
    compressed two-line certificates. Stock changes or missing cache metadata
    fall back to a full evaluation.
    """
    started=perf_counter();sweeps=tuple(sweeps);cached=previous.clipping
    processed=None if cached is None else cached.stats.get('history_count')
    stock_bounds=tuple(np.r_[low,high].tolist())
    if (processed is None or processed<0 or processed>len(sweeps) or
        cached.stats.get('stock_bounds')!=stock_bounds or cached.stats.get('backend')!=backend):
        return prepare_cut_clipping(snapshot,sweeps,low,high,backend)
    kept=np.asarray(kept)
    if kept.dtype!=np.bool_ or kept.shape!=(len(previous.endpoints['positions']),):
        raise ValueError('Clipping reuse requires a boolean mask of previous GS')
    old_count=int(kept.sum());count=len(snapshot.endpoints['positions'])
    if old_count>count:raise ValueError('Retained prefix exceeds the current GS count')
    sigma=snapshot.gaussian.attributes['scale'][:,0]
    p=snapshot.endpoints['positions'][:old_count]
    radius=CLIP_SIGMA*sigma[:old_count].astype(float)
    changed=np.zeros(count,bool);changed[old_count:]=True
    # Include pending sweeps from early-exit steps. They remove no material,
    # but the full conservative clipper can choose different certificates.
    for sweep in sweeps[processed:]:
        lo,hi=bounds(sweep)
        changed[:old_count] |= np.all((p+radius[:,None]>=lo)&(p-radius[:,None]<=hi),axis=1)
    rows=np.flatnonzero(changed);selection_ms=(perf_counter()-started)*1000
    out=np.empty((count,7),float);out[:old_count]=cached.planes[kept]
    if len(rows):
        subset=SimpleNamespace(endpoints=dict(positions=snapshot.endpoints['positions'][rows]),
            gaussian=SimpleNamespace(u=snapshot.gaussian.u[rows],v=snapshot.gaussian.v[rows],
                attributes=dict(scale=snapshot.gaussian.attributes['scale'][rows])))
        cache_started=perf_counter()
        packed=None if cutter_cache is None else cutter_cache.pack(sweeps)
        cache_ms=(perf_counter()-cache_started)*1000
        calculated=prepare_cut_clipping(subset,sweeps,low,high,backend,packed_cutters=packed)
        out[rows]=calculated.planes;stats=dict(calculated.stats)
        stats['pack_ms']+=cache_ms
    else:
        stats=dict(backend=backend,primitives=0,pack_ms=0.,kernel_ms=0. if backend=='cuda' else None,
                   history_count=len(sweeps),stock_bounds=stock_bounds)
    out.flags.writeable=False
    stats.update(total_ms=(perf_counter()-started)*1000,selection_ms=selection_ms,mode='incremental',
        input_gs=count,recomputed_gs=len(rows),reused_gs=count-len(rows),
        clipped_gs=int(np.count_nonzero(out[:,2]<CLIP_SIGMA*sigma)),
        radius_limited_gs=int(np.count_nonzero(out[:,6]<CLIP_SIGMA*sigma)))
    return ProfileClip(out,stats)


def surviving_support(planes,radius):
    """Exact nonempty test for a disc and its two linear clip constraints.

    Tests the encoded support after clipping for stock and cut-face GS. The
    closest feasible point is the origin, a projection onto a line, or the
    two-line intersection. This does not sample or subdivide the surface, nor
    independently prove that the original surface lay inside a cutter.
    """
    radius=np.minimum(np.asarray(radius),planes[:,6]);valid=radius>0
    a=planes[:,:3].copy();b=planes[:,3:6].copy()
    for q in (a,b):
        unused=~np.isfinite(q[:,2]);q[unused,:2]=0.;q[unused,2]=1.
    def feasible(x,y):
        return (x*x+y*y<=radius*radius)&(a[:,0]*x+a[:,1]*y+a[:,2]>=-1e-12)&(b[:,0]*x+b[:,1]*y+b[:,2]>=-1e-12)
    keep=feasible(0.,0.)
    for q in (a,b):keep |= feasible(-q[:,2]*q[:,0],-q[:,2]*q[:,1])
    det=a[:,0]*b[:,1]-a[:,1]*b[:,0];nonparallel=abs(det)>1e-12
    x=np.divide(a[:,1]*b[:,2]-a[:,2]*b[:,1],det,out=np.zeros(len(a)),where=nonparallel)
    y=np.divide(a[:,2]*b[:,0]-a[:,0]*b[:,2],det,out=np.zeros(len(a)),where=nonparallel)
    return valid & (keep | (nonparallel & feasible(x,y)))


def clipped_draw(snapshot, selected, size=1., clip_y=None):
    """Encode cached world-distance certificates for native two-line clipping."""
    if not np.isfinite(size) or not .25<=size<=2.:
        raise ValueError('GS display size must be between 0.25 and 2')
    result={name:value[selected].copy() for name,value in snapshot.gaussian.attributes.items()}
    planes=snapshot.clipping.planes[selected].copy()
    if clip_y is not None:
        u,v=snapshot.gaussian.u[selected],snapshot.gaussian.v[selected]
        n=np.hypot(u[:,1],v[:,1]);p=result['positions']
        d=np.divide(clip_y-p[:,1],n,out=np.full(len(p),np.inf),where=n>1e-10)
        _insert(planes,-u[:,1]/np.maximum(n,1e-30),-v[:,1]/np.maximum(n,1e-30),d)
    # Tiny/empty discs are explicitly clipped away (positive split > support).
    divisor=CLIP_SIGMA if (snapshot.clipping.stats.get('clip_subdivision') or snapshot.clipping.stats.get('clip_parent_only')) else CLIP_SIGMA+.00001
    sigma=np.maximum(1e-7,np.minimum(size*result['scale'][:,0],planes[:,6]/divisor))
    result['scale']=sigma.astype(np.float32)[:,None]
    for slot in (0,1):
        k=3*slot;d=planes[:,k+2];active=d<CLIP_SIGMA*sigma
        angle=np.arctan2(planes[:,k+1],planes[:,k]).astype(np.float32)
        angle[angle>=np.float32(np.pi)]=-np.float32(np.pi)
        result[f'clip{slot+1}_angle']=np.where(active,angle,4.).astype(np.float32)[:,None]
        result[f'clip{slot+1}_split']=np.where(active,-d/sigma,0.).astype(np.float32)[:,None]
    # Recentered residual fragments (and display-size changes) may leave
    # only the second world-space constraint active. Native GS requires
    # active constraints packed from slot one; this only reorders planes.
    second_only=(result['clip1_angle'][:,0]==4.)&(result['clip2_angle'][:,0]!=4.)
    for field in ('angle','split'):
        result[f'clip1_{field}'][second_only]=result[f'clip2_{field}'][second_only]
    result['clip2_angle'][second_only]=4.;result['clip2_split'][second_only]=0.
    empty=planes[:,6]<=0
    result['clip1_angle'][empty]=0.;result['clip1_split'][empty]=3.
    return result

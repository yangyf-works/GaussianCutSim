"""Containment certificates for a clipped tangent disk, without an outer polygon.

Linear extrema are exact disk/chord extrema. Isotropic quadratic extrema are
exact too; anisotropic quadratics use spectral and trust-region dual bounds.
Boundary contact of dimension zero/one is allowed, but a disk lying on an
external cutter face is never retired. Uncertain certificates keep the GS.
"""
import numpy as np

CUDA = r'''
// Trust-region dual bound for max(x'Px+l'x) on a disk. Any lambda above
// the largest eigenvalue is a certified upper bound; bisection merely
// tightens it. Unlike replacing P by lambda_max*I, this preserves the
// directional curvature of a cylinder projected into an oblique GS plane.
__device__ double quadratic_disk_upper(double A,double B,double C,double a,double b,double R){
 double mid=.5*(A+C),spread=hypot(.5*(A-C),B),hi=mid+spread,lo=mid-spread;
 double angle=.5*atan2(2*B,A-C),co=cos(angle),si=sin(angle);
 double h=a*co+b*si,l=-a*si+b*co;
 if(a==0.&&b==0.)return hi*R*R;
 double eps=64.*2.220446049250313e-16*(1.+fabs(hi)+hypot(a,b)/R);
 double left=hi+eps,right=left+hypot(a,b)/(2*R),best=1e300;
 for(int k=0;k<32;k++){
  double t=.5*(left+right),dh=t-hi,dl=t-lo;
  double upper=t*R*R+.25*(h*h/dh+l*l/dl);best=fmin(best,upper);
  double norm2=.25*(h*h/(dh*dh)+l*l/(dl*dl));
  if(norm2>R*R)left=t;else right=t;
 }
 return best;
}
__device__ void disk_candidate(double x,double y,double R,const double* c,double eps,double* xy,int* n){
 if(x*x+y*y>R*R+eps*(1.+R))return;
 for(int k=0;k<6;k+=3)if(isfinite(c[k+2])&&c[k]*x+c[k+1]*y+c[k+2]<-eps*(1.+hypot(c[k],c[k+1])))return;
 xy[2*(*n)]=x;xy[2*(*n)+1]=y;(*n)++;
}
__device__ int disk_candidates(double R,const double* c,double a,double b,double eps,double* xy){
 int n=0;disk_candidate(0.,0.,R,c,eps,xy,&n);
 for(int j=0;j<4;j++)disk_candidate(j==0?R:(j==1?-R:0.),j==2?R:(j==3?-R:0.),R,c,eps,xy,&n);
 double L=hypot(a,b);if(L>0.)for(int sign=-1;sign<=1;sign+=2)disk_candidate(sign*R*a/L,sign*R*b/L,R,c,eps,xy,&n);
 for(int k=0;k<6;k+=3){
  double len=hypot(c[k],c[k+1]);if(!isfinite(c[k+2])||len==0.)continue;
  double nx=c[k]/len,ny=c[k+1]/len,d=c[k+2]/len;
  if(fabs(d)<=R){double t=sqrt(fmax(0.,R*R-d*d));
   for(int sign=-1;sign<=1;sign+=2)disk_candidate(-d*nx-sign*t*ny,-d*ny+sign*t*nx,R,c,eps,xy,&n);}
 }
 double det=c[0]*c[4]-c[1]*c[3];
 if(isfinite(c[2])&&isfinite(c[5])&&fabs(det)>1e-15*hypot(c[0],c[1])*hypot(c[3],c[4]))
  disk_candidate((c[1]*c[5]-c[4]*c[2])/det,(c[3]*c[2]-c[0]*c[5])/det,R,c,eps,xy,&n);
 return n;
}
extern "C" __global__ void analytic_disk_support(const double* geometry,long long count,
 const double* records,const long long* spans,const double* planes,int parts,const double* clips,unsigned char* removed){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count||removed[i])return;
 const double* g=geometry+10*i;const double* c=clips+7*i;double R=fmin(g[9],c[6]);
 if(!(R>0.)||!isfinite(R))return;
 double eps=128.*2.220446049250313e-16*(1.+fabs(g[0])+fabs(g[1])+fabs(g[2])+R);
 for(int part=0;part<parts;part++){
  bool yes=true;
  for(long long k=spans[2*part];k<spans[2*part+1];k++){
   const double* h=planes+5*k;double a=0.,b=0.,d=h[3],xy[32];
   for(int j=0;j<3;j++){a+=h[j]*g[3+j];b+=h[j]*g[6+j];d+=h[j]*g[j];}
   int n=disk_candidates(R,c,a,b,eps,xy);if(!n){yes=false;break;}
   double lo=1e300,hi=-1e300;for(int j=0;j<n;j++){double v=a*xy[2*j]+b*xy[2*j+1]+d;lo=fmin(lo,v);hi=fmax(hi,v);}
   double tol=eps*(1.+fabs(d)+hypot(a,b));
   // A chord on the cutter boundary is not a surviving 2D face. An entire
   // coplanar disk IS a surviving face (including a flat tool's bottom).
   if(lo < -tol || (h[4] && hi<=tol)){yes=false;break;}
  }
  const double* rec=records+14*part;
  if(yes&&rec[13]){
   double delta[3],q0=0.,a=0.,b=0.,A=0.,B=0.,C=0.;for(int j=0;j<3;j++)delta[j]=g[j]-rec[j];
   for(int j=0;j<3;j++)for(int k=0;k<3;k++){
    double Q=rec[3+3*j+k];q0+=delta[j]*Q*delta[k];a+=2*g[3+j]*Q*delta[k];b+=2*g[6+j]*Q*delta[k];
    A+=g[3+j]*Q*g[3+k];B+=g[3+j]*Q*g[6+k];C+=g[6+j]*Q*g[6+k];
   }
   double xy[32];int n=disk_candidates(R,c,a,b,eps,xy);if(!n)continue;
   double lambda=.5*(A+C)+hypot(.5*(A-C),B),upper=-1e300,lower=1e300;
   // Replace the projected PSD form by lambda*I. This is an upper bound;
   // it becomes exact for a sphere in any plane and a normal cylinder cut.
   for(int j=0;j<n;j++){
    double x=xy[2*j],y=xy[2*j+1],linear=q0+a*x+b*y;
    upper=fmax(upper,linear+lambda*(x*x+y*y));
    lower=fmin(lower,linear+A*x*x+2*B*x*y+C*y*y);
   }
   double r2=rec[12]*rec[12],tol=eps*(1.+r2+fabs(q0)+R*(fabs(a)+fabs(b))+R*R*fabs(lambda));
   if(upper>r2+tol&&fabs(A-C)+fabs(B)>1e-12)
    upper=fmin(upper,q0+quadratic_disk_upper(A,B,C,a,b,R));
   if(!isfinite(upper)||upper>r2+tol||lower>=r2-tol)yes=false;
  }
  if(yes){removed[i]=1;return;}
 }
}
'''


def candidates(radius, clip, a, b, eps):
    points=[(0.,0.),(radius,0.),(-radius,0.),(0.,radius),(0.,-radius)]
    length=np.hypot(a,b)
    if length>0:points.extend([(s*radius*a/length,s*radius*b/length) for s in (-1,1)])
    for h in clip[:6].reshape(2,3):
        length=np.hypot(*h[:2])
        if not np.isfinite(h[2]) or length==0:continue
        nx,ny=h[:2]/length;d=h[2]/length
        if abs(d)<=radius:
            t=np.sqrt(max(0.,radius*radius-d*d))
            points.extend([(-d*nx-s*t*ny,-d*ny+s*t*nx) for s in (-1,1)])
    h,k=clip[:3],clip[3:6];det=h[0]*k[1]-h[1]*k[0]
    if np.isfinite(h[2]) and np.isfinite(k[2]) and abs(det)>1e-15*np.linalg.norm(h[:2])*np.linalg.norm(k[:2]):
        points.append(((h[1]*k[2]-k[1]*h[2])/det,(k[0]*h[2]-h[0]*k[2])/det))
    xy=np.asarray(points);good=(xy*xy).sum(axis=1)<=radius*radius+eps*(1+radius)
    for h in (h,k):
        if np.isfinite(h[2]):good &= xy@h[:2]+h[2]>=-eps*(1+np.linalg.norm(h[:2]))
    return xy[good]


def certify_cpu(geometry, clips, packed, removed):
    result=removed.copy()
    for i in np.flatnonzero(~result):
        g=geometry[i];c=clips[i];R=min(g[9],c[6])
        if not np.isfinite(R) or R<=0:continue
        eps=128*np.finfo(float).eps*(1+abs(g[:3]).sum()+R)
        for rec,span in zip(packed[0],packed[1]):
            yes=True
            for h in packed[2][slice(*span)]:
                a,b=h[:3]@g[3:6],h[:3]@g[6:9];d=h[:3]@g[:3]+h[3]
                xy=candidates(R,c,a,b,eps)
                if not len(xy):yes=False;break
                values=xy@np.array([a,b])+d;tol=eps*(1+abs(d)+np.hypot(a,b))
                if values.min() < -tol or (h[4] and values.max()<=tol):yes=False;break
            if yes and rec[13]:
                Q=rec[3:12].reshape(3,3);delta=g[:3]-rec[:3];basis=np.array([g[3:6],g[6:9]])
                q0=delta@Q@delta;linear=2*basis@Q@delta;P=basis@Q@basis.T
                lam=.5*np.trace(P)+np.hypot(.5*(P[0,0]-P[1,1]),P[0,1]);xy=candidates(R,c,*linear,eps)
                if not len(xy):continue
                v=q0+xy@linear;upper=np.max(v+lam*(xy*xy).sum(axis=1));lower=np.min(v+np.einsum('ni,ij,nj->n',xy,P,xy))
                r2=rec[12]**2;tol=eps*(1+r2+abs(q0)+R*abs(linear).sum()+R*R*abs(lam))
                if upper>r2+tol and abs(P[0,0]-P[1,1])+abs(P[0,1])>1e-12:
                    upper=min(upper,q0+quadratic_disk_upper_cpu(P,linear,R))
                if not np.isfinite(upper) or upper>r2+tol or lower>=r2-tol:yes=False
            if yes:result[i]=True;break
    return result


def quadratic_disk_upper_cpu(P, linear, radius):
    values,vectors=np.linalg.eigh(P);weights=(vectors.T@linear)**2
    hi=values[-1]
    if not np.any(linear):return hi*radius**2
    left=hi+64*np.finfo(float).eps*(1+abs(hi)+np.linalg.norm(linear)/radius)
    right=left+np.linalg.norm(linear)/(2*radius);best=np.inf
    for _ in range(32):
        t=.5*(left+right);d=t-values
        best=min(best,t*radius**2+.25*np.sum(weights/d))
        if .25*np.sum(weights/d**2)>radius**2:left=t
        else:right=t
    return best

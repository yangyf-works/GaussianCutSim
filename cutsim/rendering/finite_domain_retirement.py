"""Outer bounds for finite GS footprints, distinct from replacement coverage.

Intersect an enclosing polygon with inherited clipping, stock bounds and EACH
convex part's generating halfplanes. Source quadrics are deliberately omitted:
a cylinder tangent disk lies outside its source cylinder almost everywhere.
The resulting union is an OUTER bound, never evidence from empty samples.
"""
import numpy as np

CUDA = r'''
#define DOMAIN_VERTICES 80
__device__ int domain_clip(double* p,int n,double a,double b,double d){
 if(isinf(d)&&d>0.)return n;
 if(!isfinite(a)||!isfinite(b)||!isfinite(d))return -1;
 double out[2*DOMAIN_VERTICES];int count=0;
 // Expand the constraint outwards to absorb coordinate roundoff.
 d+=1e-11*(1.+fabs(d)+hypot(a,b));
 // Most finite-domain planes do not cross this small polygon. Preserve the
 // exact vertex order and arithmetic of the crossing path, but avoid local
 // scratch writes/copies when the whole polygon is admitted or excluded.
 bool inside=false,outside=false;
 for(int j=0;j<n;j++){
  double v=a*p[2*j]+b*p[2*j+1]+d;
  if(v>=0.)inside=true;else outside=true;
 }
 if(!outside)return n;
 if(!inside)return 0;
 for(int j=0;j<n;j++){
  int k=(j+1)%n;double x=p[2*j],y=p[2*j+1],xx=p[2*k],yy=p[2*k+1];
  double v=a*x+b*y+d,w=a*xx+b*yy+d;
  if(v>=0.){if(count>=DOMAIN_VERTICES)return -1;out[2*count]=x;out[2*count+1]=y;count++;}
  if((v>=0.)!=(w>=0.)){
   if(count>=DOMAIN_VERTICES)return -1;double t=v/(v-w);
   out[2*count]=x+t*(xx-x);out[2*count+1]=y+t*(yy-y);count++;
  }
 }
 for(int j=0;j<2*count;j++)p[j]=out[j];return count;
}
__device__ int finite_polygon(const double* g,int side,double R,int tile,
 const long long* span,const double* planes,const double* stock,double* p){
 int n;
 if(tile<0){
  n=32;double r=R/cos(3.141592653589793/32.);
  for(int j=0;j<n;j++){double t=j*6.283185307179586/32.;p[2*j]=r*cos(t);p[2*j+1]=r*sin(t);}
 }else{
  n=4;double h=R/32.,x=-R+(2*(tile%32)+1)*h,y=-R+(2*(tile/32)+1)*h;
  double dx=fmax(0.,fabs(x)-h),dy=fmax(0.,fabs(y)-h);if(dx*dx+dy*dy>R*R)return 0;
  p[0]=x-h;p[1]=y-h;p[2]=x+h;p[3]=y-h;p[4]=x+h;p[5]=y+h;p[6]=x-h;p[7]=y+h;
 }
 const double* c=g+13+7*side;
 for(int k=0;k<6&&n>0;k+=3)n=domain_clip(p,n,c[k],c[k+1],c[k+2]);
 for(int k=0;k<3&&n>0;k++){
  n=domain_clip(p,n,g[3+k],g[6+k],g[k]-stock[k]);
  if(n>0)n=domain_clip(p,n,-g[3+k],-g[6+k],stock[3+k]-g[k]);
 }
 for(long long k=span[0];k<span[1]&&n>0;k++){
  const double* h=planes+5*k;double a=0.,b=0.,d=h[3];
  for(int j=0;j<3;j++){a+=h[j]*g[3+j];b+=h[j]*g[6+j];d+=h[j]*g[j];}
  n=domain_clip(p,n,a,b,d);
 }
 return n;
}
__device__ bool polygon_cut(const double* g,const double* p,int n,
 const double* records,const long long* spans,const double* planes,int parts){
 if(n<3)return false;
 double area=0.;for(int j=0;j<n;j++){int k=(j+1)%n;area+=p[2*j]*p[2*k+1]-p[2*k]*p[2*j+1];}
 if(fabs(area)<1e-16)return false;
 for(int part=0;part<parts;part++){
  bool yes=true;
  for(long long k=spans[2*part];k<spans[2*part+1]&&yes;k++){
   const double* h=planes+5*k;double a=0.,b=0.,d=h[3];
   for(int j=0;j<3;j++){a+=h[j]*g[3+j];b+=h[j]*g[6+j];d+=h[j]*g[j];}
   // A complete 2D face on an external cutter plane must survive.
   double lo=1e300,hi=-1e300;
   for(int j=0;j<n;j++){double v=a*p[2*j]+b*p[2*j+1]+d;lo=fmin(lo,v);hi=fmax(hi,v);}
   double eps=1e-9*(1.+fabs(d)+hypot(a,b)*(1.+g[12]*2.));
   if(lo < -eps || (h[4]&&hi<=eps))yes=false;
  }
  const double* rec=records+14*part;
  if(yes&&rec[13]){
   // Packed quadrics are PSD. Convexity bounds the whole polygon by vertices.
   double upper=-1e300,cx=0.,cy=0.;
   for(int j=0;j<n;j++){
    double d[3],q=0.;cx+=p[2*j]/n;cy+=p[2*j+1]/n;
    for(int k=0;k<3;k++)d[k]=g[k]+p[2*j]*g[3+k]+p[2*j+1]*g[6+k]-rec[k];
    for(int k=0;k<3;k++)for(int h=0;h<3;h++)q+=d[k]*rec[3+3*k+h]*d[h];upper=fmax(upper,q);
   }
   double d[3],mid=0.;for(int k=0;k<3;k++)d[k]=g[k]+cx*g[3+k]+cy*g[6+k]-rec[k];
   for(int k=0;k<3;k++)for(int h=0;h<3;h++)mid+=d[k]*rec[3+3*k+h]*d[h];
   double r2=rec[12]*rec[12],eps=1e-9*(1.+r2+fabs(upper));
   if(!isfinite(upper)||upper>r2+eps||mid>=r2-eps)yes=false;
  }
  if(yes)return true;
 }
 return false;
}
'''


def device_domain(rt,snapshot,sweep,stock_bounds,*,rows=None):
    from cutsim.rendering.analytic_surfaces import pack_snapshot
    from cutsim.rendering.profile_cut_clip import pack_cutters
    ids,_,meta,_,spans,planes=pack_snapshot(snapshot,rows=rows)
    cut=pack_cutters((sweep,))
    cp=rt.cp
    return (cp.asarray(ids),cp.asarray(meta),cp.asarray(spans),cp.asarray(planes),
            cp.asarray(np.asarray(stock_bounds,dtype=float).reshape(6)),
            *(cp.asarray(x) for x in cut[:3]),np.int32(len(cut[0])))


def finite_cut_support(snapshot,sweep,rt,already_removed,stock_bounds,*,selection=None):
    """Certify physical removal including finite-domain limits; no hit required."""
    from cutsim.cutting.local_dexel import bounds
    from cutsim.rendering.surface_replacement import covered_rows
    cp=rt.cp;lo,hi=bounds(sweep);p=snapshot.endpoints['positions']
    r=2*snapshot.gaussian.attributes['scale'][:,0]
    with rt.context():
        excluded=cp.asnumpy(already_removed,stream=rt.stream).astype(bool)
        rows=(selection.rows[~excluded[selection.rows]] if selection is not None else
              np.flatnonzero(~excluded&np.all((p+r[:,None]>=lo)&(p-r[:,None]<=hi),axis=1)))
        result=cp.zeros(len(p),cp.uint32)
        if not len(rows):return result,0
        good=covered_rows(rt,snapshot,rows,[],sweep=sweep,stock_bounds=stock_bounds)
        result[cp.asarray(rows[good])]=1
        return result,int(good.sum())

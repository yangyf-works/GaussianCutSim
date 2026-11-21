"""Finite cutter membership and conservative analytic non-penetration proofs.

Planes, spheres and projector-form cylinders have closed-form support bounds.
A contact exemption requires separation from EVERY primitive of the peer
solid. Unsupported quadrics or uncertain comparisons never grant exemption.
"""

CUDA=r'''
__device__ bool finite_cut_inside(const double* p,const long long* m,const double* rec,
 const long long* spans,const double* planes,bool include_boundary=false){
 for(int i=m[0];i<m[0]+m[1];i++){
  bool yes=true;
  for(long long k=spans[2*i];k<spans[2*i+1];k++){
   const double* h=planes+5*k;double value=h[0]*p[0]+h[1]*p[1]+h[2]*p[2]+h[3];
   double eps=1e-9*(1.+fabs(h[3]));
   if(value < (h[4]&&!include_boundary?eps:-eps)){yes=false;break;}
  }
  const double* r=rec+14*i;
  if(yes&&r[13]){double d[3]={p[0]-r[0],p[1]-r[1],p[2]-r[2]},q=0.;
   for(int j=0;j<3;j++)for(int k=0;k<3;k++)q+=d[j]*r[3+3*j+k]*d[k];
   double eps=1e-9*(1.+r[12]*r[12]);
   if(include_boundary?q>r[12]*r[12]+eps:q>=r[12]*r[12]-eps)yes=false;
  }
  if(yes)return true;
 }
 return false;
}
__device__ bool contact_projector(const double* Q){
 double trace=Q[0]+Q[4]+Q[8];if(trace<1.99999999||trace>3.00000001)return false;
 for(int i=0;i<3;i++)for(int j=0;j<3;j++){
  double p=0.;for(int k=0;k<3;k++)p+=Q[3*i+k]*Q[3*k+j];
  if(fabs(p-Q[3*i+j])>1e-10||fabs(Q[3*i+j]-Q[3*j+i])>1e-10)return false;
 }
 return true;
}
__device__ bool surface_clear_of_solid(const double* e,const long long* m,const double* rec,
 const long long* spans,const double* planes){
 bool plane=true;for(int j=3;j<12;j++)if(e[j]!=0.)plane=false;
 double normal2=e[12]*e[12]+e[13]*e[13]+e[14]*e[14];
 if(plane&&normal2<1e-20)return false;
 if(!plane&&(normal2>1e-20||e[15]>=0.||!contact_projector(e+3)))return false;
 double radius=plane?0.:sqrt(-e[15]);
 for(int part=m[0];part<m[0]+m[1];part++){
  bool separated=false;
  for(long long k=spans[2*part];k<spans[2*part+1]&&!separated;k++){
   const double* h=planes+5*k;double h2=h[0]*h[0]+h[1]*h[1]+h[2]*h[2],value=h[3];
   for(int j=0;j<3;j++)value+=h[j]*e[j];
   double eps=1e-10*(1.+fabs(value)+radius*sqrt(h2));
   if(plane){
    double dot=h[0]*e[12]+h[1]*e[13]+h[2]*e[14];
    double cross2=fmax(0.,h2*normal2-dot*dot);
    // Equality on an internal partition is not contact with the union's
    // exterior (e.g. a ball's equator shared by its sphere and shank).
    if(cross2<=1e-20*h2*normal2&&value-e[15]*dot/normal2<=(h[4]?eps:-eps))separated=true;
   }else{
    double residual2=0.;for(int j=0;j<3;j++){double d=-h[j];for(int t=0;t<3;t++)d+=e[3+3*j+t]*h[t];residual2+=d*d;}
    if(residual2<=1e-20*fmax(1.,h2)&&value+radius*sqrt(h2)<=(h[4]?eps:-eps))separated=true;
   }
  }
  const double* r=rec+14*part;
  if(!separated&&r[13]&&contact_projector(r+3)){
   double d[3]={e[0]-r[0],e[1]-r[1],e[2]-r[2]};
   if(plane){
    double residual2=0.;for(int j=0;j<3;j++){double v=-e[12+j];for(int k=0;k<3;k++)v+=r[3+3*j+k]*e[12+k];residual2+=v*v;}
    double value=e[15];for(int j=0;j<3;j++)value-=e[12+j]*d[j];
    // Equality is tangency as well; use a scale-aware roundoff allowance.
    if(residual2<=1e-20*normal2&&fabs(value)/sqrt(normal2)>=r[12]-1e-10*(1.+r[12]))separated=true;
   }else{
    // Common projected directions give a lower bound on distance between
    // sphere/cylinder axes. Only commuting orthogonal projectors qualify.
    double P[9];bool valid=true;for(int j=0;j<3;j++)for(int k=0;k<3;k++){
     P[3*j+k]=0.;for(int t=0;t<3;t++)P[3*j+k]+=e[3+3*j+t]*r[3+3*t+k];
    }
    for(int j=0;j<3;j++)for(int k=0;k<3;k++)if(fabs(P[3*j+k]-P[3*k+j])>1e-10)valid=false;
    double dist2=0.;for(int j=0;j<3;j++){double v=0.;for(int k=0;k<3;k++)v+=P[3*j+k]*d[k];dist2+=v*v;}
    double sum=radius+r[12];if(valid&&dist2>=sum*sum-1e-10*(1.+sum*sum))separated=true;
   }
  }
  if(!separated)return false;
 }
 return true;
}
'''

"""Continuous planar floors use their finite generating domain, not an old cap."""
import numpy as np

CUDA = r'''
__device__ bool floor_domain_inside(const long long* m,const double* p,const double* rec,
 const long long* spans,const double* planes){
 for(int i=m[0];i<m[0]+m[1];i++){
  bool yes=true;for(long long k=spans[2*i];k<spans[2*i+1];k++){
   const double* h=planes+5*k;double value=h[0]*p[0]+h[1]*p[1]+h[2]*p[2]+h[3];
   if(value < -1e-9*(1.+fabs(h[3]))){yes=false;break;}
  }
  const double* r=rec+14*i;
  if(yes&&r[13]){double d[3]={p[0]-r[0],p[1]-r[1],p[2]-r[2]},q=0.;
   for(int j=0;j<3;j++)for(int k=0;k<3;k++)q+=d[j]*r[3+3*j+k]*d[k];
   if(q>r[12]*r[12]+1e-9*(1.+r[12]*r[12]))yes=false;
  }
  if(yes)return true;
 }
 return false;
}
__device__ bool physical_planar_floor(const double* e,const long long* m,const double* p,
 const long long* spans,const double* planes){
 if(m[3])return false;for(int j=3;j<12;j++)if(e[j]!=0.)return false;
 double norm=e[12]*e[12]+e[13]*e[13]+e[14]*e[14];if(norm==0.)return false;
 for(long long k=spans[2*m[2]];k<spans[2*m[2]+1];k++){
  const double* h=planes+5*k;if(!h[4])continue;
  double cross[3]={h[1]*e[14]-h[2]*e[13],h[2]*e[12]-h[0]*e[14],h[0]*e[13]-h[1]*e[12]};
  double value=h[0]*p[0]+h[1]*p[1]+h[2]*p[2]+h[3];
  if(fabs(value)<=1e-8*(1.+fabs(h[3]))&&cross[0]*cross[0]+cross[1]*cross[1]+cross[2]*cross[2]<=1e-18*norm*(h[0]*h[0]+h[1]*h[1]+h[2]*h[2]))return true;
 }
 return false;
}
'''


def extend_round_floors(snapshot,nold,draw,material):
    """Lift a round plunge floor into its same-ID capsule extension.

    This is restricted to a matched endpoint cylinder of a flat capsule.
    Its circular cross-section is contained in that capsule by construction.
    Only old circle tangents are released; other cut boundaries stay inherited.
    The GPU subsequently evaluates the whole finite capsule domain.
    """
    table=snapshot.analytic_patches;ids=snapshot.analytic_patch_ids.copy()
    original=ids.copy();surfaces=snapshot.material_surface_ids;changed=0
    fresh=[]
    for key in np.unique(ids[nold:]):
        patch=table[int(key)];eq=patch.equation;rec=patch.solid.packed[0]
        if patch.stock or np.any(eq[3:14]) or not eq[14] or len(rec)!=3 or rec[patch.part,13]:continue
        # Exactly one straight prism and its two circular endpoint pieces.
        circles=rec[rec[:,13]!=0]
        if len(circles)!=2 or not np.allclose(circles[:,3:12],np.diag([1.,1.,0.]).reshape(1,9),atol=1e-12,rtol=0):continue
        fresh.append((int(key),patch,circles,np.unique(surfaces[nold:][ids[nold:]==key])))
    for key in np.unique(original[:nold]):
        old=table[int(key)];eq=old.equation;rec=old.solid.packed[0]
        if old.stock or len(rec)!=1 or not rec[0,13] or np.any(eq[3:14]) or not eq[14]:continue
        if not np.allclose(rec[0,3:12],np.diag([1.,1.,0.]).ravel(),atol=1e-12,rtol=0):continue
        for newkey,new,circles,allowed in fresh:
            if not np.allclose(eq,new.equation,atol=1e-9,rtol=0):continue
            matching=np.linalg.norm(circles[:,:2]-rec[0,:2],axis=1)<1e-9
            matching &= abs(circles[:,12]-rec[0,12])<1e-9
            if not matching.any():continue
            rows=np.flatnonzero((original[:nold]==key)&np.isin(surfaces[:nold],allowed))
            for row in rows:
                p=snapshot.endpoints['positions'][row];centre=rec[0,:3]-p
                local=np.array([centre@snapshot.gaussian.u[row],centre@snapshot.gaussian.v[row]])
                radius=2*float(snapshot.gaussian.attributes['scale'][row,0]);released=False
                for clips in (draw,material):
                    for slot in (0,3):
                        h=clips[row,slot:slot+3];ideal=rec[0,12]-h[:2]@local
                        # An old tangent may already be the exact new side
                        # wall. Preserve that boundary instead of releasing it.
                        exact_wall=False
                        packed=new.solid.packed
                        for wall in packed[2][slice(*packed[1][new.part])]:
                            if not wall[4]:continue
                            normal=np.array([wall[:3]@snapshot.gaussian.u[row],wall[:3]@snapshot.gaussian.v[row]])
                            norm=np.linalg.norm(normal)
                            if norm>1e-12:
                                projected=np.r_[normal/norm,(wall[:3]@p+wall[3])/norm]
                                if np.allclose(h,projected,atol=1e-9,rtol=0):exact_wall=True;break
                        if exact_wall:continue
                        if np.isfinite(h[2]) and abs(np.linalg.norm(h[:2])-1.)<1e-8 and abs(h[2]-ideal)<=.061*radius+1e-8:
                            clips[row,slot:slot+3]=[0.,0.,np.inf];released=True
                if released:ids[row]=newkey;changed+=1
    if changed:
        from cutsim.rendering.analytic_surfaces import set_metadata
        set_metadata(snapshot,ids,table)
    return changed

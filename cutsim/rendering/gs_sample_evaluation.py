"""Shared local quadrature and CUDA evaluation used by R and clipping.

The centre opacity cancels in normalized errors/R. Coordinates are physical
distances in the GS tangent frame; sigma is never the UI display multiplier.
"""
import numpy as np


def disc(radial, angular):
    r = np.sqrt((np.arange(radial) + .5) / radial)
    theta = (np.arange(angular) + .5) * 2 * np.pi / angular
    return np.ascontiguousarray(np.stack((r[:, None] * np.cos(theta),
        r[:, None] * np.sin(theta)), axis=-1).reshape(-1, 2))


def device_disc(rt, radial=16, angular=32):
    """Share immutable CUDA quadrature between R and clipping on one runtime."""
    if not hasattr(rt, 'gs_quadrature_tables'):
        rt.gs_quadrature_tables = {}
    key = (radial, angular)
    if key not in rt.gs_quadrature_tables:
        rt.gs_quadrature_tables[key] = rt.cp.asarray(disc(*key))
    return rt.gs_quadrature_tables[key]


CUDA = r'''
__device__ void gs_sample_point(const double* g,double x,double y,double* p){
 for(int k=0;k<3;k++)p[k]=g[k]+x*g[3+k]+y*g[6+k];
}
__device__ bool gs_sample_admitted(const double* c,double x,double y){
 return c[0]*x+c[1]*y+c[2]>=0. && c[3]*x+c[4]*y+c[5]>=0.;
}
__device__ double gs_sample_weight(double x,double y,double sigma){
 return exp(-.5*(x*x+y*y)/(sigma*sigma));
}
__device__ double gs_quadratic_form(const double* Q,const double* d){
 double q=0.;for(int a=0;a<3;a++)for(int b=0;b<3;b++)q+=d[a]*Q[3*a+b]*d[b];return q;
}
__device__ double gs_quadric(const double* e,const double* p,double* grad){
 double d[3]={p[0]-e[0],p[1]-e[1],p[2]-e[2]},f=e[15];
 for(int a=0;a<3;a++){
  double q=0.;for(int b=0;b<3;b++)q+=e[3+3*a+b]*d[b];
  grad[a]=2*q+e[12+a];f+=d[a]*(q+e[12+a]);
 }
 return f;
}
'''

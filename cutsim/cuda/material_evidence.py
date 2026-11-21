"""Shared BEFORE material evidence for base rays and children.

Positive evidence comes only from supported finite analytic faces. Retained
finite cutting solids can reject already-cut space, but cannot prove material.
The evaluator is the same cached evaluator used by stage 2.
"""
import numpy as np


CUT_REJECTION_CUDA = r'''
extern "C" __global__ void coverage_cut_rejection(const double* points,int count,
 const long long* solids,int nsolids,const double* rec,const long long* spans,
 const double* planes,signed char* states,unsigned char* rejected){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count||states[i]<0)return;
 for(int j=0;j<nsolids;j++){
  const long long m[4]={solids[2*j],solids[2*j+1],0,0};
  // A retained generating cutter certifies removed space even if its local
  // GS coverage is missing. Boundary/contact is not newly cut material.
  if(finite_cut_inside(points+3*i,m,rec,spans,planes,true)){
   states[i]=-1;rejected[i]=1;return;
  }
 }
}
'''


def finite_material_evidence_device(surface, points, directions):
    """Recheck material using the prepared finite analytic face selection.

    Reuse the BEFORE evaluator prepared by generation. A published AFTER
    surface cannot certify what should have been generated: missing GS would
    otherwise erase their own witnesses. Equations keep their live footprints
    and finite domains; unused historical solids are never unioned into material.
    Selected generating solids may reject already-cut space beyond their GS footprint; their
    absence never proves material. Unknown remains unverified.
    """
    from cutsim.cuda.analytic_ray_generation import evaluator
    from cutsim.rendering.finite_surface_contact import CUDA as contact_cuda
    field = evaluator(surface)
    rt = surface.rt; cp = rt.cp
    with rt.context():
        points = cp.ascontiguousarray(cp.asarray(points, dtype=cp.float64))
        directions = cp.ascontiguousarray(cp.asarray(directions, dtype=cp.float64))
        if directions.shape != points.shape:
            raise ValueError('One direction required per analytic query')
        states = cp.zeros(len(points), cp.int8)
        fallback = cp.zeros(len(points), cp.uint8)
        solids = field.coverage_cut_solids
        rejected = cp.zeros(len(states), cp.uint8)
        if len(states) and len(solids):
            if not hasattr(rt, 'coverage_cut_rejection_kernel'):
                rt.coverage_cut_rejection_kernel = cp.RawKernel(
                    contact_cuda + CUT_REJECTION_CUDA, 'coverage_cut_rejection', options=('--fmad=false',))
            rt.coverage_cut_rejection_kernel(((len(states)+127)//128,), (128,),
                (points, np.int32(len(states)),
                 solids, np.int32(len(solids)), *field.data[3:], states, rejected))
            # Cut membership is an unconditional veto, so resolve it before
            # expensive bidirectional face searches. The finite-face predicate
            # still handles every remaining point; absence of a cutter is
            # never evidence of material. Keep all original probe rows.
            remaining = cp.flatnonzero(states == 0)
            states[remaining], _, fallback[remaining] = field.evaluate(points[remaining], directions[remaining])
        else:
            states, _, fallback = field.evaluate(points, directions)
    return states, fallback, rejected


def finite_material_evidence(surface, points, directions):
    """CPU result adapter; production base rays keep classification on GPU."""
    from cutsim.cuda.analytic_ray_generation import evaluator
    field = evaluator(surface)
    states, _, rejected = finite_material_evidence_device(surface, points, directions)
    with surface.rt.context():
        rejections = int(surface.rt.cp.count_nonzero(rejected).item())
        states = surface._download(states)
    return states, dict(material_method='finite_analytic_faces',
        analytic_faces=len(field.data[1]),
        analytic_cut_rejections=rejections,
        analytic_material_points=int(np.count_nonzero(states == 1)),
        analytic_air_points=int(np.count_nonzero(states < 0)),
        analytic_unknown_points=int(np.count_nonzero(states == 0)))

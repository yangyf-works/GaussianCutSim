"""Bounded final-surface diagnostics; never authorize retirement or create GS.

Samples are witnesses, not a watertightness proof. Unexamined source rows,
unsupported charts and ambiguous material must remain visible in the report.
"""
from dataclasses import dataclass, replace
import numpy as np
from cutsim.rendering.opaque_coverage import (
    DEFAULT_DISPLAY_SCALE, REFERENCE_ALPHA, OPAQUE_TRANSMITTANCE,
    SURFEL_CUTOFF_SIGMA, opaque_geometry,
)


@dataclass(frozen=True)
class CoverageProbes:
    points: np.ndarray
    normals: np.ndarray
    expected: np.ndarray  # N x 2: draw, material
    tolerance: np.ndarray
    source_rows: np.ndarray  # BEFORE row, or -1 for a tool cell
    tool_rows: np.ndarray  # generator-owned probe-grid row, or -1
    curvature: float = 0.


@dataclass(frozen=True)
class CoverageReport:
    probes: CoverageProbes
    missing: np.ndarray  # N x 2, independently checked domains
    unchecked: dict
    display_scale: float = DEFAULT_DISPLAY_SCALE

    @property
    def repair_rows(self):
        take = self.missing.any(axis=1) & (self.probes.tool_rows >= 0)
        return np.unique(self.probes.tool_rows[take])

    @property
    def status(self):
        if self.missing.any():
            return 'missing'
        return 'unverified' if any(self.unchecked.values()) else 'sampled_ok'

    def stats(self):
        old = self.probes.source_rows >= 0
        return dict(coverage_status=self.status, coverage_probes=len(self.missing),
            coverage_draw_method='opaque_surface_opacity',
            coverage_display_scale=float(self.display_scale),
            coverage_reference_alpha=REFERENCE_ALPHA,
            coverage_transmittance_threshold=OPAQUE_TRANSMITTANCE,
            coverage_expected_draw=int(self.probes.expected[:, 0].sum()),
            coverage_expected_material=int(self.probes.expected[:, 1].sum()),
            coverage_missing_draw=int(self.missing[:, 0].sum()),
            coverage_missing_material=int(self.missing[:, 1].sum()),
            coverage_lost_old=int(self.missing[old].any(axis=1).sum()),
            coverage_missing_tool=int(self.missing[~old].any(axis=1).sum()),
            **{'coverage_unchecked_' + k: int(v) for k, v in self.unchecked.items()})

    def packet(self, limit=256):
        """Small replay/UI diagnostic, not the GPU work grid or a repair engine."""
        rows = np.flatnonzero(self.missing.any(axis=1))
        kept = rows[:limit]
        return dict(stats=self.stats(), points=self.probes.points[kept].copy(),
            normals=self.probes.normals[kept].copy(), missing=self.missing[kept].copy(),
            source_rows=self.probes.source_rows[kept].copy(),
            tool_rows=self.probes.tool_rows[kept].copy(), omitted=len(rows)-len(kept),
            scope='local_before_support_and_current_tool_cells')


CUDA = r'''
__device__ void coverage_at(const double* p,const double* n,double tolerance,
 long long source,const double* gs,int ng,double factor,double curvature,
 double alpha,double threshold,double cutoff,const double* boxes,int chunk,unsigned char* result){
 result[0]=result[1]=0;
 double transmittance=1.;
 for(int j=0;j<ng;j++){
  // Contiguous groups preserve the original opacity multiplication order.
  if(chunk && j%chunk==0){
   const double* b=boxes+6*(j/chunk);
   if(p[0]<b[0]||p[1]<b[1]||p[2]<b[2]||p[0]>b[3]||p[1]>b[4]||p[2]>b[5]){
    j+=chunk-1;continue;
   }
  }
  const double* g=gs+27*j;double u=0.,v=0.,h=0.,dot=0.;
  for(int k=0;k<3;k++){
   double d=p[k]-g[k];u+=d*g[3+k];v+=d*g[6+k];h+=d*g[9+k];dot+=n[k]*g[9+k];
  }
  double tol=tolerance;
  if(source<0 && curvature>0.)tol+=fmin(.05,4.*g[12]*g[12]/(2.*curvature));
  if(dot<(source<0?.9:.9961946980917455)||fabs(h)>tol)continue;
  for(int side=0;side<2;side++){
   if(result[side])continue;
   const double* c=g+13+7*side;
   double r=side?fmin(factor*g[12],c[6]):cutoff*c[6];
   if(!(r>0.)||u*u+v*v>r*r+1e-12)continue;
   if(c[0]*u+c[1]*v+c[2]<-1e-9||c[3]*u+c[4]*v+c[5]<-1e-9)continue;
   if(side)result[1]=1;
   else{
    transmittance*=1.-alpha*exp(-.5*(u*u+v*v)/(c[6]*c[6]));
    result[0]=transmittance<threshold;
   }
  }
  if(result[0]&&result[1])return;
 }
}
extern "C" __global__ void coverage_points(const double* points,const double* normals,
 const double* tolerance,const long long* source,const unsigned char* expected,int count,const double* gs,int ng,
 const double* before,int nb,int has_before,double factor,double curvature,
 double alpha,double threshold,double cutoff,const double* boxes,const double* before_boxes,
 int chunk,unsigned char* result){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 result[4*i]=result[4*i+1]=0;
 result[4*i+2]=result[4*i+3]=1;
 // Air/unknown probes cannot request either coverage domain or a repair.
 if(!expected[2*i]&&!expected[2*i+1])return;
 coverage_at(points+3*i,normals+3*i,tolerance[i],source[i],gs,ng,factor,curvature,
             alpha,threshold,cutoff,boxes,chunk,result+4*i);
 if(has_before && source[i]>=0)
  coverage_at(points+3*i,normals+3*i,tolerance[i],source[i],before,nb,factor,curvature,
              alpha,threshold,cutoff,before_boxes,chunk,result+4*i+2);
}
'''


def _coverage_boxes(geometry, factor, margin, chunk):
    """Conservative bounds for consecutive GS, without sorting opacity donors.

    The Gram matrix lower bound also handles imperfect local frames. If it
    cannot bound a frame, keep that group unbounded (fall back to scanning).
    Include both the radial comparison epsilon and maximum sheet tolerance.
    """
    if not len(geometry) or not chunk:
        return np.empty((0, 6), float)
    radius = np.maximum(SURFEL_CUTOFF_SIGMA*geometry[:, 19],
                        np.minimum(factor*geometry[:, 12], geometry[:, 26]))
    radius = np.maximum(radius, 0.)
    frame = geometry[:, 3:12].reshape(-1, 3, 3)
    gram = frame @ frame.transpose(0, 2, 1)
    diagonal = np.diagonal(gram, axis1=1, axis2=2)
    lower = np.min(diagonal - (np.abs(gram).sum(axis=2)-np.abs(diagonal)), axis=1)
    reach = np.full(len(geometry), np.inf)
    valid = np.isfinite(lower) & (lower > 1e-12)
    reach[valid] = np.sqrt((radius[valid]**2+1e-12+margin**2)/lower[valid])
    # Outward padding for FP64 projection/bound arithmetic, not a new predicate.
    reach += 1e-8*(1.+np.abs(geometry[:, :3]).max(axis=1)+reach)
    starts = np.arange(0, len(geometry), chunk)
    return np.ascontiguousarray(np.column_stack((
        np.minimum.reduceat(geometry[:, :3]-reach[:, None], starts),
        np.maximum.reduceat(geometry[:, :3]+reach[:, None], starts))))


def audit_coverage(rt, snapshot, probes, unchecked=None, *,
                   display_scale=DEFAULT_DISPLAY_SCALE, before=None,
                   before_draw=None, spatial_index=True):
    """One audit of opaque opacity/material, optionally protecting BEFORE opacity.

This is a local surface test, not a camera/pixel test. Incompatible sheets
cannot contribute opacity to conceal a missing machining face.
before_draw, when supplied, must be opaque_geometry(before, display_scale)
from this same finalization. spatial_index=False is the dense reference path.
"""
    from cutsim.rendering.gs_extent import support_factor
    cp = rt.cp
    n = len(probes.points)
    if not n:
        return CoverageReport(probes, np.zeros((0, 2), bool), dict(unchecked or {}), display_scale)
    margin = .05 + float(np.max(probes.tolerance, initial=0.))
    low, high = probes.points.min(axis=0)-margin, probes.points.max(axis=0)+margin
    def packed(state, draw=None):
        p = state.endpoints['positions']; g = state.gaussian
        sigma = g.attributes['scale'][:, 0]
        if draw is None:
            draw = opaque_geometry(state, display_scale)
        reach = np.maximum(SURFEL_CUTOFF_SIGMA*draw[:, 6], support_factor(state)*sigma)
        ids = np.flatnonzero(np.all((p+reach[:, None]>=low)&(p-reach[:, None]<=high), axis=1))
        return np.ascontiguousarray(np.column_stack((p[ids], g.u[ids], g.v[ids],
            state.endpoints['normals'][ids], sigma[ids], draw[ids],
            state.material_clipping.planes[ids])), dtype=np.float64)
    geometry = packed(snapshot)
    previous = packed(before, before_draw) if before is not None else np.empty((0, 27))
    chunk = 32 if spatial_index else 0
    factor = support_factor(snapshot)
    boxes = _coverage_boxes(geometry, factor, margin, chunk)
    before_boxes = _coverage_boxes(previous, factor, margin, chunk)
    with rt.context():
        if not hasattr(rt, 'coverage_points_kernel'):
            rt.coverage_points_kernel = cp.RawKernel(CUDA, 'coverage_points', options=('--fmad=false',))
        result = cp.empty((n, 4), cp.uint8)
        rt.coverage_points_kernel(((n+127)//128,), (128,),
            (cp.asarray(np.ascontiguousarray(probes.points, dtype=float)),
             cp.asarray(np.ascontiguousarray(probes.normals, dtype=float)),
             cp.asarray(probes.tolerance), cp.asarray(probes.source_rows),
             cp.asarray(np.ascontiguousarray(probes.expected, dtype=np.uint8)), np.int32(n),
              cp.asarray(geometry), np.int32(len(geometry)), cp.asarray(previous),
              np.int32(len(previous)), np.int32(before is not None), np.float64(support_factor(snapshot)),
              np.float64(probes.curvature), np.float64(REFERENCE_ALPHA),
              np.float64(OPAQUE_TRANSMITTANCE), np.float64(SURFEL_CUTOFF_SIGMA),
              cp.asarray(boxes), cp.asarray(before_boxes), np.int32(chunk), result))
        covered = cp.asnumpy(result, stream=rt.stream).astype(bool)
    expected = probes.expected.copy()
    old = probes.source_rows >= 0
    expected[old, 0] &= covered[old, 2]
    probes = replace(probes, expected=expected)
    return CoverageReport(probes, expected & ~covered[:, :2], dict(unchecked or {}), display_scale)


def retained_probes(before, sweep, stock_bounds, max_probes=4096, *,
                    display_scale=DEFAULT_DISPLAY_SCALE, draw=None):
    """Witness local BEFORE support outside the current cutter, in both domains.

Does not infer surfaces in existing holes from nearest-plane signs. A whole
source with no GS has no witness here; the report explicitly scopes this audit.
"""
    from cutsim.cutting.local_dexel import bounds
    from cutsim.rendering.gs_extent import support_factor
    p = before.endpoints['positions']; g = before.gaussian
    sigma = g.attributes['scale'][:, 0].astype(float)
    low, high = bounds(sweep)
    if draw is None:
        draw = opaque_geometry(before, display_scale)
    draw_radius = SURFEL_CUTOFF_SIGMA*draw[:, 6]
    reach = np.maximum(draw_radius, support_factor(before)*sigma)
    local = np.flatnonzero(np.all((p+reach[:, None]>=low)&(p-reach[:, None]<=high), axis=1))
    axis = np.linspace(-1., 1., 5)
    x, y = np.meshgrid(axis, axis)
    disc = np.column_stack((x.ravel(), y.ravel()))
    disc = disc[(disc*disc).sum(axis=1)<=1.]
    count = min(len(local), max_probes//(2*len(disc)))
    ids = local[np.linspace(0, len(local)-1, count, dtype=np.int64)] if count else local[:0]
    sources = np.repeat(ids, 2*len(disc))
    uv = np.concatenate((draw_radius[ids, None, None]*disc,
                         support_factor(before)*sigma[ids, None, None]*disc), axis=1).reshape(-1, 2)
    points = p[sources]+uv[:, :1]*g.u[sources]+uv[:, 1:]*g.v[sources]
    expected = np.empty((len(points), 2), bool)
    for side in (0, 1):
        c = (draw if side == 0 else before.material_clipping.planes)[sources]
        r = draw_radius[sources] if side == 0 else np.minimum(support_factor(before)*sigma[sources], c[:, 6])
        yes = (r>0)&((uv*uv).sum(axis=1)<=r*r+1e-12)
        for k in (0, 3):
            yes &= (uv*c[:, k:k+2]).sum(axis=1)+c[:, k+2]>=-1e-9
        expected[:, side] = yes
    valid = expected.any(axis=1)
    valid &= np.all((points>=stock_bounds[0])&(points<=stock_bounds[1]), axis=1)
    if len(points):
        valid &= sweep.sdf(points)>=-1e-9
    return CoverageProbes(points[valid], before.endpoints['normals'][sources[valid]], expected[valid],
        np.minimum(.005, .04*sigma[sources[valid]]), sources[valid],
        np.full(int(valid.sum()), -1, np.int64)), len(local)-len(ids)


def join_probes(old, tool):
    return CoverageProbes(*(np.concatenate((getattr(old, name), getattr(tool, name)))
        for name in ('points', 'normals', 'expected', 'tolerance', 'source_rows', 'tool_rows')),
        curvature=tool.curvature)

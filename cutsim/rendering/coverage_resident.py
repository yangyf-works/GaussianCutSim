"""Bounded CUDA-resident child pruning; only scores, near-ties and masks return.

Probe generation stays on CPU. Dense bounded device matrices avoid constructing
and transferring a CPU sparse graph. The same finite support and clip evaluator
is shared with the reference CUDA backend; greedy child order is preserved.
"""
from time import perf_counter
import os
from threading import BoundedSemaphore, local
import numpy as np

from cutsim.rendering.coverage_cuda import KERNEL, packed_inputs


RESIDENT_KERNEL = r'''
extern "C" __global__ void resident_matrix(
    const double* projected, const double* centres, const float* scale,
    const float* positions, const float* world32, const float* u32, const float* v32,
    const double* angles, const float* splits, const double* cosine, const double* sine,
    const float* cosine32, const float* sine32, int children, int probes, int dual, float* output) {
    int k = blockIdx.x*blockDim.x+threadIdx.x;
    int rows = probes*(dual ? 2 : 1);
    if (k >= children*rows) return;
    int i=k%rows, j=k/rows;
    output[k] = attenuation(projected, centres, scale, positions, world32, u32, v32,
        angles, splits, cosine, sine, cosine32, sine32, i%probes, j, i>=probes);
}

__device__ double coverage_required(double reference, bool boundary, double tolerance) {
    double allowed = fmin(boundary ? fmin(.01,tolerance) : tolerance, .1*reference);
    double target = reference-allowed;
    if (reference>=.5) target=fmax(target,fmin(reference,.75));
    if (boundary && reference>=.25 && reference<.75) target=reference;
    return -log1p(-target);
}

extern "C" __global__ void resident_totals(const float* matrix, const unsigned char* boundary,
    int children, int probes, int rows, double tolerance, const unsigned char* keep,
    double* full, double* current, double* required) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=rows) return;
    double a=0., b=0.;
    for (int j=0;j<children;j++) {
        double v=matrix[j*rows+i];
        a+=v;
        if (keep[j]) b+=v;
    }
    full[i]=a; current[i]=b;
    required[i]=coverage_required(-expm1(-a),boundary[i%probes]!=0,tolerance);
}

extern "C" __global__ void resident_impacts(const float* matrix, const double* full,
    int probes, double* output) {
    __shared__ double sums[256];
    int j=blockIdx.x, t=threadIdx.x;
    double value=0.;
    for (int i=t;i<probes;i+=256) value+=exp(-full[i])*expm1((double)matrix[j*probes+i]);
    sums[t]=value;
    __syncthreads();
    for (int offset=128;offset;offset>>=1) {
        if (t<offset) sums[t]+=sums[t+offset];
        __syncthreads();
    }
    if (!t) output[j]=sums[0];
}

extern "C" __global__ void resident_remove(const float* matrix, int children, int probes,
    const int* order, double* current, const double* required, unsigned char* keep, int* counts) {
    int t=threadIdx.x;
    if (!t) counts[0]=0;
    for (int k=0;k<children;k++) {
        int j=order[k];
        bool fail=false;
        for (int i=t;i<probes;i+=256)
            if (current[i]-matrix[j*probes+i]<required[i]-1e-10) fail=true;
        if (!__syncthreads_or(fail)) {
            if (!t) {keep[j]=0; counts[0]++;}
            for (int i=t;i<probes;i+=256) current[i]-=matrix[j*probes+i];
        }
        __syncthreads();
    }
}

// Stable maximum: the earliest probe/child wins an equal value, like np.argmax.
__device__ void stable_max(double value, int index, double* values, int* indices) {
    int t=threadIdx.x;
    values[t]=value; indices[t]=index;
    __syncthreads();
    for (int offset=128;offset;offset>>=1) {
        if (t<offset && (values[t+offset]>values[t] ||
            (values[t+offset]==values[t] && indices[t+offset]<indices[t]))) {
            values[t]=values[t+offset]; indices[t]=indices[t+offset];
        }
        __syncthreads();
    }
}

extern "C" __global__ void resident_restore(const float* matrix, int children, int probes,
    double* current, const double* required, unsigned char* keep, int* counts) {
    __shared__ double values[256];
    __shared__ int indices[256];
    int t=threadIdx.x;
    if (!t) counts[1]=0;
    for (int step=0;step<children;step++) {
        double maximum=-1e300; int index=2147483647;
        for (int i=t;i<probes;i+=256) {
            double value=required[i]-current[i];
            if (value>maximum) {maximum=value; index=i;}
        }
        stable_max(maximum,index,values,indices);
        if (values[0]<=1e-10) return;
        int worst=indices[0];
        __syncthreads();
        maximum=-1.; index=2147483647;
        for (int j=t;j<children;j+=256) if (!keep[j]) {
            double value=matrix[j*probes+worst];
            if (value>maximum) {maximum=value; index=j;}
        }
        stable_max(maximum,index,values,indices);
        if (indices[0]==2147483647) return;
        int chosen=indices[0];
        __syncthreads();
        if (!t) {keep[chosen]=1; counts[1]++;}
        for (int i=t;i<probes;i+=256) current[i]+=matrix[chosen*probes+i];
        __syncthreads();
    }
}

extern "C" __global__ void resident_diagnostics(
    const float* decision, const double* decision_full, const unsigned char* decision_edge, int dp,
    const double* validation_full, const double* validation_current, const unsigned char* validation_edge,
    int vp, int children, const unsigned char* keep, const int* counts, double* output) {
    __shared__ double drops[256], edges[256];
    __shared__ int holes[256];
    int t=threadIdx.x, count=0;
    double maximum=0., boundary=0.;
    for (int i=t;i<dp+2*vp;i+=256) {
        double full, current; bool edge;
        if (i<dp) {
            full=decision_full[i]; current=0.; edge=decision_edge[i]!=0;
            for (int j=0;j<children;j++) if (keep[j]) current+=decision[j*dp+i];
        } else {
            int k=i-dp;
            full=validation_full[k]; current=validation_current[k]; edge=validation_edge[k%vp]!=0;
        }
        double before=-expm1(-full), after=-expm1(-current), drop=before-after;
        maximum=fmax(maximum,drop);
        if (edge) boundary=fmax(boundary,drop);
        count+=(before>=.5 && after<.5);
    }
    drops[t]=maximum; edges[t]=boundary; holes[t]=count;
    __syncthreads();
    for (int offset=128;offset;offset>>=1) {
        if (t<offset) {
            drops[t]=fmax(drops[t],drops[t+offset]);
            edges[t]=fmax(edges[t],edges[t+offset]); holes[t]+=holes[t+offset];
        }
        __syncthreads();
    }
    if (!t) {
        output[0]=drops[0]; output[1]=edges[0]; output[2]=holes[0];
        output[3]=counts[0]; output[4]=counts[1];
    }
    for (int j=t;j<children;j+=256) output[5+j]=keep[j];
}
'''


class ResidentPruner:
    def __init__(self, gpu):
        self.gpu = gpu
        count = int(os.environ.get('CUTSIM_RESIDENT_SLOTS', '4'))
        if not 1 <= count <= 8:
            raise ValueError('CUTSIM_RESIDENT_SLOTS must be between 1 and 8')
        self.slots = BoundedSemaphore(count)
        self.slot_count = count
        self.local = local()
        self.pool = gpu.cp.cuda.MemoryPool()
        self.pool.set_limit(size=count*160*1024*1024)
        names = ('matrix', 'totals', 'impacts', 'remove', 'restore', 'diagnostics')
        self.kernels = {name:gpu.cp.RawKernel(KERNEL+RESIDENT_KERNEL, 'resident_'+name,
                                            options=('--fmad=false',)) for name in names}
        for kernel in self.kernels.values():
            kernel.compile()

    def prune(self, draw, points, boundary, checks, check_boundary, origin, basis, tolerance, stats):
        gpu, cp = self.gpu, self.gpu.cp
        c, p, q = len(draw['positions']), len(points), len(checks)
        if c > 512 or not p or not q or c*(p+2*q) > gpu.MAX_PAIRS:
            raise ValueError('Resident coverage budget exceeded')
        # Preserve CPU projection and trigonometric conventions, including the
        # separately rounded world points used for Float32 clip validation.
        a = packed_inputs(draw, points, origin, basis)
        b = packed_inputs(draw, checks, origin, basis)
        with self.slots, cp.cuda.Device(gpu.device), cp.cuda.using_allocator(self.pool.malloc):
            if not hasattr(self.local, 'stream'):
                self.local.stream = cp.cuda.Stream(non_blocking=True)
            stream = self.local.stream
            try:
                return self._execute(a,b,boundary,check_boundary,c,p,q,tolerance,stats,stream)
            except BaseException:
                stream.synchronize()
                raise

    def _execute(self, a, b, boundary, check_boundary, c, p, q, tolerance, stats, stream):
        gpu, cp = self.gpu, self.gpu.cp
        with stream:
            started = perf_counter()
            arrays = [cp.asarray(x) for x in a]
            validation_arrays = arrays.copy()
            for i in (0, 4):
                validation_arrays[i] = cp.asarray(b[i])
            edge, check_edge = cp.asarray(boundary), cp.asarray(check_boundary)
            decision = cp.empty((c, p), cp.float32)
            validation = cp.empty((c, 2*q), cp.float32)
            keep = cp.ones(c, cp.uint8)
            counts = cp.zeros(2, cp.int32)
            full, current, required = [cp.empty(p, cp.float64) for _ in range(3)]
            check_full, check_current, check_required = [cp.empty(2*q, cp.float64) for _ in range(3)]
            scores = cp.empty(c, cp.float64)
            metrics = cp.empty(5+c, cp.float64)
            def launch(name, size, args):
                self.kernels[name](((size+255)//256,), (256,), args)
            ints = lambda *values: tuple(np.int32(v) for v in values)
            launch('matrix', c*p, (*arrays, *ints(c,p,0), decision))
            launch('totals', p, (decision, edge, *ints(c,p,p), np.float64(tolerance), keep, full, current, required))
            self.kernels['impacts']((c,), (256,), (decision, full, np.int32(p), scores))
            impact = cp.asnumpy(scores, stream=stream)
            readback = impact.nbytes
            order = np.argsort(impact, kind='stable')
            error = 16.*np.finfo(float).eps*(p+1)*np.abs(impact)
            close = np.diff(impact[order]) <= error[order[:-1]]+error[order[1:]]
            refine = np.unique(np.r_[order[:-1][close], order[1:][close]])
            if len(refine):
                columns = cp.asnumpy(decision[cp.asarray(refine)], stream=stream)
                totals = cp.asnumpy(full, stream=stream)
                readback += columns.nbytes+totals.nbytes
                transmission = np.exp(-totals)
                for j, values in zip(refine, columns):
                    impact[j] = np.sum(transmission*np.expm1(values.astype(float)))
                order = np.argsort(impact, kind='stable')
            device_order = cp.asarray(order.astype(np.int32))
            self.kernels['remove']((1,), (256,), (decision, *ints(c,p), device_order, current, required, keep, counts))
            launch('matrix', c*2*q, (*validation_arrays, *ints(c,q,1), validation))
            launch('totals', 2*q, (validation, check_edge, *ints(c,q,2*q), np.float64(tolerance),
                                  keep, check_full, check_current, check_required))
            self.kernels['restore']((1,), (256,), (validation, *ints(c,2*q), check_current, check_required, keep, counts))
            self.kernels['diagnostics']((1,), (256,), (decision, full, edge, np.int32(p), check_full,
                check_current, check_edge, *ints(q,c), keep, counts, metrics))
            measured = cp.asnumpy(metrics, stream=stream)
            mask, removed = measured[5:].astype(bool), measured[3:5]
            readback += measured.nbytes
            elapsed = perf_counter()-started
            with gpu.lock:
                gpu.calls += 1
                gpu.resident_calls += 1
                gpu.pairs += c*(p+2*q)
                gpu.readback_bytes += readback
                gpu.service_s += elapsed
        stats.update(coverage_algorithm='resident', order_refined_children=len(refine),
                     resident_readback_bytes=readback, resident_matrix_pairs=c*(p+2*q))
        if removed[0]:
            stats.update(validation_probes=2*q, restored=int(removed[1]), max_drop=float(measured[0]),
                         boundary_max_drop=float(measured[1]), new_holes=int(measured[2]))
        return np.flatnonzero(mask)


def prune(*args):
    from cutsim.rendering.coverage_cuda import evaluator
    gpu = evaluator()
    if gpu is None:
        raise ValueError('Resident coverage requires CUTSIM_COVERAGE_BACKEND=cuda')
    with gpu.lock, gpu.cp.cuda.Device(gpu.device):
        if gpu.resident is None:
            gpu.resident = ResidentPruner(gpu)
    return gpu.resident.prune(*args)

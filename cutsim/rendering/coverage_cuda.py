"""Optional CUDA coverage evaluation for the existing draw-child pruning rules.

The CPU still generates probes and performs ordered deletion/restoration. Only
the dense probe x child attenuation matrix is evaluated here. No CUDA dependency
is imported until CUTSIM_COVERAGE_BACKEND=cuda is explicitly selected.
"""
import os
from threading import Lock
from time import perf_counter

import numpy as np


KERNEL = r'''
__device__ float attenuation(
    const double* projected, const double* centres, const float* scale,
    const float* positions, const float* world32, const float* u32, const float* v32,
    const double* angles, const float* splits, const double* cosine, const double* sine,
    const float* cosine32, const float* sine32, const int i, const int j, const int use_float32) {
    double x = (projected[2*i]-centres[2*j])/scale[j];
    double y = (projected[2*i+1]-centres[2*j+1])/scale[j];
    double rho = x*x+y*y;
    bool keep = rho <= 9.;
    float cx = 0, cy = 0;
    if (use_float32) {
        float dx = world32[3*i]-positions[3*j];
        float dy = world32[3*i+1]-positions[3*j+1];
        float dz = world32[3*i+2]-positions[3*j+2];
        cx = ((dx*u32[3*j]+dy*u32[3*j+1])+dz*u32[3*j+2])/scale[j];
        cy = ((dx*v32[3*j]+dy*v32[3*j+1])+dz*v32[3*j+2])/scale[j];
    }
    for (int edge=0;edge<2;edge++) {
        int at=2*j+edge;
        if (use_float32) {
            float value=cx*cosine32[at]+cy*sine32[at];
            keep=keep && (angles[at]==4. || value>=splits[at]);
        } else {
            double value=x*cosine[at]+y*sine[at];
            keep=keep && (angles[at]==4. || value>=splits[at]);
        }
    }
    double alpha=keep ? .99*exp(-.5*rho) : 0.;
    return (float)(-log1p(-alpha));
}

extern "C" __global__ void coverage(
    const double* projected, const double* centres, const float* scale,
    const float* positions, const float* world32, const float* u32, const float* v32,
    const double* angles, const float* splits, const double* cosine, const double* sine,
    const float* cosine32, const float* sine32, const int children,
    const int probes, const int use_float32, float* output) {
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= (unsigned long long)children * probes) return;
    output[k] = attenuation(projected, centres, scale, positions, world32, u32, v32,
        angles, splits, cosine, sine, cosine32, sine32, k/children, k%children, use_float32);
}

extern "C" __global__ void coverage_pairs(
    const double* projected, const double* centres, const float* scale,
    const float* positions, const float* world32, const float* u32, const float* v32,
    const double* angles, const float* splits, const double* cosine, const double* sine,
    const float* cosine32, const float* sine32, const int* probes, const int* children,
    const int pairs, const int dual, float* output) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= pairs) return;
    output[k] = attenuation(projected, centres, scale, positions, world32, u32, v32,
        angles, splits, cosine, sine, cosine32, sine32, probes[k], children[k], 0);
    if (dual) output[k+pairs] = attenuation(projected, centres, scale, positions, world32, u32, v32,
        angles, splits, cosine, sine, cosine32, sine32, probes[k], children[k], 1);
}
'''


def backend_key():
    """Also used in exact fallback-cache keys; never alias CPU/CUDA entries."""
    name = os.environ.get('CUTSIM_COVERAGE_BACKEND', 'cpu').lower()
    if name not in ('cpu', 'cuda'):
        raise ValueError('CUTSIM_COVERAGE_BACKEND must be cpu or cuda')
    device = int(os.environ.get('CUTSIM_COVERAGE_DEVICE', '0')) if name == 'cuda' else None
    if device is not None and device < 0:
        raise ValueError('CUTSIM_COVERAGE_DEVICE must be a nonnegative CUDA device index')
    from cutsim.rendering.coverage_sparse import algorithm
    return name, device, 1, algorithm()


def packed_inputs(draw, points, origin, basis):
    """Match the CPU projection and Float32 validation operation order."""
    from cutsim.rendering.gaussian_attributes import _render_axes
    u, v = _render_axes(draw['rot'])
    world = origin + points @ basis.T
    projected = np.column_stack((world @ u[0], world @ v[0]))
    centres = np.column_stack((np.sum(draw['positions']*u, axis=1),
                               np.sum(draw['positions']*v, axis=1)))
    angles = np.column_stack((draw['clip1_angle'][:, 0], draw['clip2_angle'][:, 0])).astype(float)
    splits = np.column_stack((draw['clip1_split'][:, 0], draw['clip2_split'][:, 0]))
    arrays = [projected, centres, draw['scale'][:, 0], draw['positions'],
              world, u, v, angles, splits, np.cos(angles), np.sin(angles),
              np.cos(angles.astype(np.float32)), np.sin(angles.astype(np.float32))]
    # RawKernel does not check pointer dtypes. Make the ABI explicit, including
    # callers with Float32 probe coordinates or noncontiguous child attributes.
    doubles = {0, 1, 7, 9, 10}
    return [np.ascontiguousarray(a, dtype=np.float64 if i in doubles else np.float32)
            for i, a in enumerate(arrays)]


class CudaCoverage:
    """One bounded stream per device, shared safely by parent worker threads."""
    MAX_PAIRS = 16_000_000  # Same guard as the pruning caller; <= 64 MB output.

    def __init__(self, device):
        started = perf_counter()
        from cutsim.core.cuda_cache import configure_cuda_cache
        configure_cuda_cache()
        try:
            import cupy as cp
        except ImportError as error:
            raise RuntimeError('CUDA coverage requires CuPy; see requirements.txt') from error
        self.cp, self.device, self.lock = cp, device, Lock()
        with cp.cuda.Device(device):
            self.pool = cp.cuda.MemoryPool()
            self.pool.set_limit(size=256*1024*1024)
            self.stream = cp.cuda.Stream(non_blocking=True)
            self.kernel = cp.RawKernel(KERNEL, 'coverage', options=('--fmad=false',))
            self.kernel.compile()
            self.pair_kernel = cp.RawKernel(KERNEL, 'coverage_pairs', options=('--fmad=false',))
            self.pair_kernel.compile()
            self.gpu_name = cp.cuda.runtime.getDeviceProperties(device)['name'].decode()
        self.output = None
        self.calls = self.pairs = 0
        self.readback_bytes = self.resident_calls = 0
        self.resident = None
        self.service_s = 0.
        self.initialization_s = perf_counter()-started

    def evaluate(self, draw, points, origin, basis, float32_clips=False):
        probes, children = len(points), len(draw['positions'])
        pairs = probes*children
        if pairs > self.MAX_PAIRS:
            raise ValueError('CUDA coverage probe budget exceeded; split the work explicitly')
        if not pairs:
            return np.empty((probes, children), np.float32)
        arrays = packed_inputs(draw, points, origin, basis)
        cp = self.cp
        # Keep only one matrix on the device even when many CPU parents run.
        # The returned NumPy array owns its storage; no worker sees reused memory.
        with self.lock, cp.cuda.Device(self.device), self.stream, cp.cuda.using_allocator(self.pool.malloc):
            started = perf_counter()
            if self.output is None or self.output.size < pairs:
                self.output = None
                self.pool.free_all_blocks()
                self.output = cp.empty(pairs, dtype=cp.float32)
            device = [cp.asarray(a) for a in arrays]
            output = self.output[:pairs].reshape(probes, children)
            self.kernel(((pairs+255)//256,), (256,),
                        (*device, np.int32(children), np.int32(probes), np.int32(float32_clips), output))
            result = cp.asnumpy(output, stream=self.stream, blocking=True)
            self.calls += 1
            self.pairs += pairs
            self.readback_bytes += result.nbytes
            self.service_s += perf_counter()-started
            return result

    def stats(self):
        with self.lock:
            return dict(device=self.device, gpu=self.gpu_name, calls=self.calls, pairs=self.pairs,
                        readback_bytes=self.readback_bytes, resident_calls=self.resident_calls, service_s=self.service_s,
                        initialization_s=self.initialization_s, pool_bytes=self.pool.total_bytes(),
                        resident_pool_bytes=self.resident.pool.total_bytes() if self.resident is not None else 0,
                        resident_slots=self.resident.slot_count if self.resident is not None else 0)

    def evaluate_pairs(self, arrays, pointers, indices, dual):
        """Read back only finite-support neighbours; share validation uploads."""
        pairs = len(indices)
        entries = pairs*(2 if dual else 1)
        if entries > self.MAX_PAIRS:
            raise ValueError('CUDA coverage probe budget exceeded; split the work explicitly')
        if not pairs:
            empty = np.empty(0, np.float32)
            return empty, empty
        columns = np.repeat(np.arange(len(pointers)-1, dtype=np.int32), np.diff(pointers))
        cp = self.cp
        with self.lock, cp.cuda.Device(self.device), self.stream, cp.cuda.using_allocator(self.pool.malloc):
            started = perf_counter()
            if self.output is None or self.output.size < entries:
                self.output = None
                self.pool.free_all_blocks()
                self.output = cp.empty(entries, dtype=cp.float32)
            device = [cp.asarray(a) for a in arrays]
            rows, children = cp.asarray(indices), cp.asarray(columns)
            output = self.output[:entries]
            self.pair_kernel(((pairs+255)//256,), (256,),
                             (*device, rows, children, np.int32(pairs), np.int32(dual), output))
            result = cp.asnumpy(output, stream=self.stream, blocking=True)
            self.calls += 1
            self.pairs += entries
            self.readback_bytes += result.nbytes
            self.service_s += perf_counter()-started
            return result[:pairs], result[pairs:] if dual else result[:pairs]


_lock = Lock()
_evaluators = {}


def evaluator():
    name, device, *_ = backend_key()
    if name == 'cpu':
        return None
    with _lock:
        if device not in _evaluators:
            _evaluators[device] = CudaCoverage(device)
        return _evaluators[device]

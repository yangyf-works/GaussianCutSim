"""Shared CUDA device, stream, allocator and runtime ownership."""
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from time import perf_counter
import numpy as np

class ProfileCudaRuntime:
    def __init__(self, device=0):
        started = perf_counter()
        if not isinstance(device, (int, np.integer)) or device < 0:
            raise ValueError('Nonnegative CUDA device index required')
        from cutsim.core.cuda_cache import configure_cuda_cache
        configure_cuda_cache()
        try:
            import cupy as cp
        except ImportError as error:
            raise RuntimeError('CUDA profile dexel requires requirements.txt') from error
        self.cp, self.device, self.lock = cp, int(device), RLock()
        with cp.cuda.Device(self.device):
            self.stream = cp.cuda.Stream(non_blocking=True)
            self.pool = cp.cuda.MemoryPool()
            self.pool.set_limit(size=512*1024*1024)
            self.name = cp.cuda.runtime.getDeviceProperties(self.device)['name'].decode()
        self.initialization_s = perf_counter()-started

    @contextmanager
    def context(self):
        cp = self.cp
        with self.lock, cp.cuda.Device(self.device), self.stream, cp.cuda.using_allocator(self.pool.malloc):
            try:
                yield
            except BaseException:
                self.stream.synchronize()
                raise



_runtimes = {}
_runtime_lock = RLock()


def runtime(device=0):
    with _runtime_lock:
        if device not in _runtimes:
            _runtimes[device] = ProfileCudaRuntime(device)
        return _runtimes[device]

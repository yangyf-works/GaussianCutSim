"""Bounded reuse of pure draw-domain calculations across CPU preparations.

Keys contain exact bytes, shapes and dtypes, never rounded geometry or a
material-history approximation. Results are shared read-only by convention;
callers attach private diagnostic dictionaries, not mutable cache entries.
"""
from collections import OrderedDict
from threading import RLock
import numpy as np


def array_key(value):
    value = np.asarray(value)
    return value.dtype.str, value.shape, value.tobytes()


def owned_bytes(value):
    if isinstance(value, np.ndarray): return 128+value.nbytes
    if isinstance(value, bytes): return 64+len(value)
    if isinstance(value, dict): return 256+sum(owned_bytes(k)+owned_bytes(v) for k,v in value.items())
    if isinstance(value, (tuple, list)): return 64+sum(map(owned_bytes,value))
    return 64


def freeze_arrays(value):
    if isinstance(value, np.ndarray): value.flags.writeable = False
    elif isinstance(value, dict):
        for item in value.values(): freeze_arrays(item)
    elif isinstance(value, (tuple, list)):
        for item in value: freeze_arrays(item)


class ExactDrawCache:
    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = 0
        self.lock = RLock()

    def get(self, key):
        with self.lock:
            entry = self.entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self.hits += 1
            self.entries.move_to_end(key)
            return entry[0]

    def put(self, key, value):
        size = owned_bytes(key)+owned_bytes(value)
        if size > self.max_bytes: return
        freeze_arrays(value)
        with self.lock:
            old = self.entries.pop(key, None)
            if old is not None: self.bytes -= old[1]
            while self.entries and self.bytes+size > self.max_bytes:
                _, (_, removed) = self.entries.popitem(last=False)
                self.bytes -= removed
            self.entries[key] = value, size
            self.bytes += size

    def clear(self):
        with self.lock:
            self.entries.clear()
            self.bytes = self.hits = self.misses = 0


domain_cache = ExactDrawCache(32*1024*1024)
fallback_cache = ExactDrawCache(64*1024*1024)

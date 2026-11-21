"""Immutable stable-ID buckets shared by geometry, rendering and replay storage.

Buckets are storage units, not spatial cells: they change no geometric tolerance.
They own their buffers and contain no references to previous block states.
"""
import numpy as np

BUCKET_SIZE = 1024


def freeze(value):
    if isinstance(value, np.ndarray):
        value.flags.writeable = False
        if value.dtype.hasobject:
            for item in value.flat:
                freeze(item)
    elif isinstance(value, dict):
        for item in value.values():
            freeze(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            freeze(item)


class FrozenRows:
    """Private, owned row buffers. Never edit after publication."""
    def __init__(self, ids, arrays, diagnostics=None):
        self.ids = np.array(ids, dtype=np.int64, copy=True)
        self.arrays = {name: np.array(value, copy=True) for name, value in arrays.items()}
        self.diagnostics = {} if diagnostics is None else diagnostics
        self._freeze()

    def _freeze(self):
        freeze(self.ids); freeze(self.arrays); freeze(self.diagnostics)

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._freeze()


def bucket_ranges(ids):
    """Contiguous ranges of ascending stable IDs; no per-row Python loop."""
    keys = np.asarray(ids, np.int64) // BUCKET_SIZE
    bounds = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1])+1, len(keys)]
    return [(int(keys[a]), int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]

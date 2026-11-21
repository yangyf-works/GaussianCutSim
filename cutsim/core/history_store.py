"""Private, lossless disk backing for histories; only locally written pickles.

Hot entries are bounded by count. Files live as long as this owner, including
while playback holds the owner. No precision conversion or geometry evaluation.
"""
from collections import OrderedDict
from collections.abc import MutableMapping, Sequence
from pathlib import Path
import pickle
import tempfile
import os


def history_enabled():
    return os.environ.get("CUTSIM_DISK_HISTORY", "0")=="1"


class HistoryMap(MutableMapping):
    def __init__(self, keep=2):
        self.keep=keep
        self._directory=tempfile.TemporaryDirectory(prefix='cutsim-history-',
            dir=os.environ.get('CUTSIM_HISTORY_DIR'))
        self._paths={}; self._hot=OrderedDict(); self._serial=0

    def __len__(self):return len(self._paths)
    def __iter__(self):return iter(self._paths)
    def __getitem__(self,key):
        if key not in self._hot:
            with self._paths[key].open('rb') as stream:self._hot[key]=pickle.load(stream)
        self._hot.move_to_end(key)
        value=self._hot[key]
        while len(self._hot)>self.keep:self._hot.popitem(last=False)
        return value

    def __setitem__(self,key,value):
        path=Path(self._directory.name)/str(self._serial);self._serial+=1
        try:
            with path.open('wb') as stream:pickle.dump(value,stream,protocol=5)
        except BaseException:
            path.unlink(missing_ok=True);raise
        old=self._paths.get(key);self._paths[key]=path
        if old is not None:old.unlink(missing_ok=True)
        self._hot[key]=value;self._hot.move_to_end(key)
        while len(self._hot)>self.keep:self._hot.popitem(last=False)

    def __delitem__(self,key):
        path=self._paths.pop(key);self._hot.pop(key,None);path.unlink(missing_ok=True)


class HistorySequence(Sequence):
    def __init__(self, values=()):
        self.store=HistoryMap()
        for value in values:self.append(value)
    def __len__(self):return len(self.store)
    def append(self,value):self.store[len(self)]=value
    def __getitem__(self,index):
        if isinstance(index,slice):return [self[i] for i in range(*index.indices(len(self)))]
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        return self.store[index]


def checkpoint_geometry(sim, interval=32, *, render_cache=None):
    """New geometry root; analytic material history and stable IDs stay intact.

    Existing snapshots are not mutated. A cache at this exact endpoint can be
    rebound without refitting unchanged GS. Other caches rebuild normally.
    """
    from copy import copy
    from cutsim.core.local_state import _LocalCloudState
    from cutsim.core.splat_cloud import SplatCloud
    old=sim.splats._local_state
    if old.depth<interval:return False
    cloud=copy(sim.splats)
    cloud.__dict__.pop('_local_state',None)
    root=_LocalCloudState(None,SplatCloud.empty(),cloud)
    root.cell_size=old.root.cell_size
    sim.splats=copy(cloud);sim.splats._local_state=root
    if render_cache is not None:
        render_cache.rebase_geometry(old,root)
    return True

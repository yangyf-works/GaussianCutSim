"""Lossless replay snapshots with shared compressed array chunks.

One writer (CPU worker), immutable manifests and one append-only blob file.
Readers need no hash index and never write from the UI thread. Array chunks are
content addressed: a new snapshot writes only chunks not already in the session.
No precision conversion, quantization or geometry evaluation occurs here.
"""
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import gzip
import os
import pickle
import tempfile
import zlib
import weakref
import numpy as np
from cutsim.core.frozen_rows import FrozenRows

# Weak references share unchanged immutable buckets across loaded adjacent
# blocks without pinning the entire replay history in the GUI process.
_decoded_rows = weakref.WeakValueDictionary()


def replay_archive_enabled():
    return os.environ.get('CUTSIM_REPLAY_ARCHIVE', '1') != '0'


class ArchiveOwner:
    def __init__(self):
        self.directory = tempfile.TemporaryDirectory(prefix='cutsim-replay-',
            dir=os.environ.get('CUTSIM_HISTORY_DIR'))
        self.path = self.directory.name


@dataclass
class ReplayReference:
    directory: str
    name: str
    part: str = 'block'
    owner: object = field(default=None, repr=False, compare=False)

    def select(self, part):
        return ReplayReference(self.directory, self.name, part, self.owner)

    def load(self):
        with (Path(self.directory)/'arrays.bin').open('rb') as blobs:
            with gzip.open(Path(self.directory)/self.name,'rb') as stream:
                block = _Reader(stream, blobs).load()
        block.delta._archive_reference = self
        block.delta._archive_endpoints = block.attributes
        object.__setattr__(block.attributes, '_archive_reference', self)
        if self.part == 'delta': return block.delta
        if self.part == 'prepared':
            return (block.delta, block.attributes.before, block.attributes.after)
        return block


class _Reader(pickle.Unpickler):
    def __init__(self, stream, blobs):
        super().__init__(stream); self.blobs=blobs; self.arrays={}

    def persistent_load(self, description):
        if description[0] == 'rows-v1':
            _,number,record = description
            if number in self.arrays:return self.arrays[number]
            offset,size,raw_size=record
            key=(str(self.blobs.name),offset,size,raw_size)
            result=_decoded_rows.get(key)
            if result is None:
                self.blobs.seek(offset)
                data=zlib.decompress(self.blobs.read(size))
                if len(data)!=raw_size:raise IOError('Invalid replay row bucket length')
                result=pickle.loads(data)
                if not isinstance(result,FrozenRows):raise pickle.UnpicklingError('Invalid replay rows')
                _decoded_rows[key]=result
            self.arrays[number]=result
            return result
        tag, number, dtype, shape, writable, chunks = description
        if tag != 'array-v1': raise pickle.UnpicklingError('Unknown replay record')
        if number in self.arrays: return self.arrays[number]
        result=np.empty(shape, dtype=dtype)
        target=memoryview(result).cast('B');cursor=0
        for offset, size, raw_size in chunks:
            self.blobs.seek(offset)
            data=zlib.decompress(self.blobs.read(size))
            if len(data)!=raw_size: raise IOError('Invalid replay chunk length')
            target[cursor:cursor+raw_size]=data;cursor+=raw_size
        if cursor != result.nbytes: raise IOError('Incomplete replay array')
        result.flags.writeable=writable
        self.arrays[number]=result
        return result


class ReplayArchive:
    chunk_size=256*1024

    def __init__(self, directory):
        self.directory=Path(directory);self.index={};self.serial=0
        self.logical_bytes=0;self.unique_bytes=0;self.stored_bytes=0
        self.row_records=weakref.WeakKeyDictionary()
        self.row_hits=0;self.row_writes=0;self.row_reused_bytes=0
        self.directory.mkdir(parents=True,exist_ok=True)
        (self.directory/'arrays.bin').touch()

    def write(self, block):
        memo={};archive=self;layouts={}
        # Use stable-ID buckets as chunk boundaries for GS row arrays. Deleting
        # a row must not shift every subsequent chunk and defeat sharing.
        # Row order itself is unchanged; readers concatenate the original order.
        def register(values, ids):
            if ids is None or not len(ids):return
            buckets=np.asarray(ids,dtype=np.int64)//1024
            boundaries=np.r_[0,np.flatnonzero(buckets[1:]!=buckets[:-1])+1,len(ids)]
            for value in values:
                if isinstance(value,np.ndarray) and value.ndim and len(value)==len(ids):
                    layouts[id(value)]=(value,boundaries)
        from cutsim.core.splat_cloud import SplatCloud
        for cloud in vars(block.delta).values():
            if isinstance(cloud,SplatCloud) and not hasattr(cloud,'chunks'):
                register(vars(cloud).values(),getattr(cloud,'_local_ids',None))
        for name, cloud_name in (('before','before'),('after','_local_final_cloud')):
            cloud=getattr(block.delta,cloud_name,None)
            endpoint=getattr(block.attributes,name,None)
            if cloud is not None and isinstance(endpoint,dict) and not getattr(endpoint,'_immutable_chunks',False):
                register(endpoint.values(),getattr(cloud,'_local_ids',None))
        with (self.directory/'arrays.bin').open('ab') as blobs:
            class Writer(pickle.Pickler):
                def persistent_id(self, value):
                    if isinstance(value,FrozenRows):
                        identity=id(value)
                        if identity in memo:return memo[identity][1]
                        record=archive.row_records.get(value)
                        if record is None:
                            data=pickle.dumps(value,protocol=5)
                            digest=hashlib.sha256(data).digest()
                            record=archive.index.get(digest)
                            if record is None:
                                packed=zlib.compress(data,1)
                                record=(blobs.tell(),len(packed),len(data))
                                blobs.write(packed);archive.index[digest]=record
                                archive.unique_bytes+=len(data);archive.stored_bytes+=len(packed)
                            archive.row_records[value]=record
                            archive.row_writes+=1
                        else:
                            archive.row_hits+=1;archive.row_reused_bytes+=record[2]
                        archive.logical_bytes+=record[2]
                        result=('rows-v1',len(memo),record)
                        memo[identity]=(value,result)
                        return result
                    if not isinstance(value,np.ndarray) or value.dtype.hasobject or value.nbytes<4096:
                        return None
                    identity=id(value)
                    if identity in memo:return memo[identity][1]
                    view=memoryview(np.ascontiguousarray(value)).cast('B')
                    chunks=[]
                    layout=layouts.get(identity)
                    if layout is None:
                        boundaries=range(0,len(view)+archive.chunk_size,archive.chunk_size)
                    else:
                        row_bytes=value.nbytes//len(value)
                        boundaries=layout[1]*row_bytes
                    for start,end in zip(boundaries[:-1],boundaries[1:]):
                        data=view[int(start):int(end)]
                        digest=hashlib.sha256(data).digest()
                        archive.logical_bytes+=len(data)
                        record=archive.index.get(digest)
                        if record is None:
                            packed=zlib.compress(data,1)
                            record=(blobs.tell(),len(packed),len(data))
                            blobs.write(packed);archive.index[digest]=record
                            archive.unique_bytes+=len(data);archive.stored_bytes+=len(packed)
                        chunks.append(record)
                    result=('array-v1',len(memo),value.dtype,value.shape,
                            bool(value.flags.writeable),tuple(chunks))
                    memo[identity]=(value,result)
                    return result
            name=f'block-{self.serial:06d}.pickle.gz'
            temporary=self.directory/(name+'.tmp')
            with gzip.open(temporary,'wb',compresslevel=1) as stream:
                Writer(stream,protocol=5).dump(block)
            blobs.flush()
        temporary.replace(self.directory/name)
        self.serial+=1
        return ReplayReference(str(self.directory),name)

    def metrics(self):
        return dict(logical_array_bytes=self.logical_bytes,unique_array_bytes=self.unique_bytes,
                    compressed_array_bytes=self.stored_bytes,unique_chunks=len(self.index),
                    immutable_bucket_hits=self.row_hits,immutable_bucket_writes=self.row_writes,
                    immutable_reused_bytes=self.row_reused_bytes,
                    disk_bytes=sum(p.stat().st_size for p in self.directory.iterdir() if p.is_file()))


class ReferenceMap(dict):
    """Small in-memory index; at most two decoded historical blocks stay hot."""
    def __init__(self, keep=2):
        super().__init__();self.hot=OrderedDict();self.keep=keep

    def adopt(self,key,reference,value=None):
        super().__setitem__(key,reference)
        if value is not None:self.hot[key]=value
        else:self.hot.pop(key,None)
        self._trim()

    def _trim(self):
        while len(self.hot)>self.keep:
            key,_=self.hot.popitem(last=False)
            # Recomputed display settings have no archived endpoint yet.
            # Drop only that derived value; the geometric history still owns
            # its immutable archive reference and can recompute it on demand.
            if not isinstance(super().get(key),ReplayReference):
                super().__delitem__(key)

    def cache_value(self,key,value):
        super().__setitem__(key,value)
        self.hot[key]=value
        self.hot.move_to_end(key)
        self._trim()

    def __getitem__(self,key):
        value=super().__getitem__(key)
        if not isinstance(value,ReplayReference):return value
        if key not in self.hot:self.hot[key]=value.load()
        self.hot.move_to_end(key);self._trim()
        return self.hot[key]

    def get(self,key,default=None):
        return self[key] if key in self else default

    def clear(self):
        super().clear();self.hot.clear()


class ReplaySequence(Sequence):
    def __init__(self, values=()):
        self.store=ReferenceMap()
        for value in values:self.append(value)
    def __len__(self):return len(self.store)
    def __getitem__(self,index):
        if isinstance(index,slice):return [self[i] for i in range(*index.indices(len(self)))]
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        return self.store[index]
    def append(self,delta):
        ref=getattr(delta,'_archive_reference',None)
        if ref is None:self.store[len(self)]=delta
        else:self.store.adopt(len(self),ref.select('delta'),delta)
    def archived(self,ref):
        # Called after append by the sole CPU writer. The raw delta (including
        # its computation work graph) is no longer retained by the history.
        self.store.adopt(len(self)-1,ref.select('delta'))

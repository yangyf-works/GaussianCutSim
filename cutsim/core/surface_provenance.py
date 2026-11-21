"""Immutable stable-ID face identities, shared with all replay prefixes."""
from collections.abc import Mapping
import numpy as np
from cutsim.core.frozen_rows import FrozenRows,BUCKET_SIZE,bucket_ranges


class SurfaceProvenance(Mapping):
    def __init__(self,chunks=None,count=0):
        self.chunks={} if chunks is None else chunks
        self.count=count

    def __len__(self):return self.count

    def __iter__(self):
        for key in sorted(self.chunks):
            chunk=self.chunks[key]
            yield from map(int,chunk.ids[chunk.arrays['identity']!=0])

    def __getitem__(self,key):
        key=int(key);chunk=self.chunks.get(key//BUCKET_SIZE)
        if chunk is None:raise KeyError(key)
        value=int(chunk.arrays['identity'][key%BUCKET_SIZE])
        if not value:raise KeyError(key)
        return value

    def lookup(self, ids):
        """Read exact identities in row order, including absent/retired IDs."""
        ids=np.asarray(ids,np.int64)
        result=np.zeros(ids.shape,np.int32)
        order=np.argsort(ids,kind='stable')
        sorted_ids=ids[order]
        for key,a,b in bucket_ranges(sorted_ids):
            chunk=self.chunks.get(key)
            if chunk is not None:
                result[order[a:b]]=chunk.arrays['identity'][sorted_ids[a:b]%BUCKET_SIZE]
        return result

    @classmethod
    def updated(cls,previous,ids,values):
        if not isinstance(previous,cls):
            original=previous or {}
            base=cls()
            if original:
                old_ids=np.asarray(sorted(original),np.int64)
                base=cls.updated(base,old_ids,np.asarray([original[int(i)] for i in old_ids],np.int32))
        else:base=previous
        chunks=base.chunks.copy();count=base.count
        ids=np.asarray(ids,np.int64);values=np.asarray(values,np.int32)
        order=np.argsort(ids,kind='stable');ids,values=ids[order],values[order]
        for key,a,b in bucket_ranges(ids):
            old=chunks.get(key);data=np.zeros(BUCKET_SIZE,np.int32) if old is None else old.arrays['identity'].copy()
            rows=ids[a:b]%BUCKET_SIZE
            count+=int(np.count_nonzero(values[a:b]))-int(np.count_nonzero(data[rows]))
            data[rows]=values[a:b]
            chunks[key]=FrozenRows(np.arange(key*BUCKET_SIZE,(key+1)*BUCKET_SIZE,dtype=np.int64),{'identity':data})
        return cls(chunks,count)

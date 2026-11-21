"""Transient ordered edge batches; no per-GS Python lists or normalization.

Rows are CPU publication indices. Dense edge blocks are uploaded and packed
on CUDA together at final clipping. Nothing is stored in published snapshots.
"""
import numpy as np


class EdgeBatches:
    def __init__(self,blocks=()):
        self.blocks=list(blocks)

    def append(self,rows,edges):
        rows=np.asarray(rows,np.int64)
        if len(rows):self.blocks.append((rows,np.asarray(edges,float).reshape(len(rows),-1,3)))

    def keys(self):
        return np.unique(np.concatenate([r for r,_ in self.blocks])) if self.blocks else np.empty(0,np.int64)

    def __bool__(self):return bool(self.blocks)

    def remap(self,mapping):
        result=EdgeBatches()
        for rows,edges in self.blocks:
            mapped=mapping[rows];keep=mapped>=0
            if keep.any():result.append(mapped[keep],edges[keep])
        return result

    def select(self,rows,total):
        if len(np.unique(rows))!=len(rows):
            # Rare debug selections may duplicate a source; keep its line order.
            result=EdgeBatches()
            for output,source in enumerate(rows):
                for r,e in self.blocks:
                    match=r==source
                    if match.any():result.append([output],e[match].reshape(1,-1,3))
            return result
        mapping=np.full(total,-1,np.int64);mapping[rows]=np.arange(len(rows))
        return self.remap(mapping)

    def joined(self,other,offset):
        return EdgeBatches(self.blocks+[(rows+offset,edges) for rows,edges in other.blocks])

    def center_mask(self,count):
        from cutsim.cuda.profile_dexel_cuda import runtime
        rt=runtime();cp=rt.cp
        with rt.context():
            keep=cp.ones(count,cp.int32)
            for rows,edges in self.blocks:
                values=cp.all(cp.asarray(edges)[:,:,2]>=-1e-9,axis=1).astype(cp.int32)
                cp.minimum.at(keep,cp.asarray(rows),values)
            return cp.asnumpy(keep,stream=rt.stream).astype(bool)


def batches(raw):
    if isinstance(raw,EdgeBatches):return raw
    result=EdgeBatches()
    for row,edges in (raw or {}).items():result.append([row],np.asarray(edges).reshape(1,-1,3))
    return result

"""Finite GS/GS edges must belong to both surviving CPU surface regions."""
import numpy as np
from cutsim.geometry.strip_kernels import njit, enabled


@njit(cache=True, nogil=True)
def _trim_segments(segments, domain_offsets, plane_offsets, planes):
    maximum=0
    own_count=domain_offsets[1]-domain_offsets[0]
    for j in range(len(segments)):
        maximum+=own_count*(domain_offsets[j+2]-domain_offsets[j+1])
    indices=np.empty(maximum,np.int64)
    output=np.empty((maximum,2,3))
    intervals=np.empty((maximum,2))
    count=0
    for j in range(len(segments)):
        start=segments[j,0];delta=segments[j,1]-start
        length=np.sqrt(np.sum(delta*delta))
        collar=64.*np.finfo(np.float64).eps*(1.+np.max(np.abs(segments[j])))
        min_fraction=collar/max(length,1e-30)
        number=0
        for own in range(domain_offsets[0],domain_offsets[1]):
            for peer in range(domain_offsets[j+1],domain_offsets[j+2]):
                low=0.;high=1.;valid=True
                for branch in (own,peer):
                    for p in range(plane_offsets[branch],plane_offsets[branch+1]):
                        value=planes[p,3]
                        slope=0.;norm=0.
                        for axis in range(3):
                            value+=planes[p,axis]*start[axis]
                            slope+=planes[p,axis]*delta[axis]
                            norm+=planes[p,axis]*planes[p,axis]
                        norm=np.sqrt(norm)
                        if abs(slope)<=collar*max(norm,1e-30):
                            if value < -max(1e-10,collar)*max(norm,1e-30):
                                valid=False;break
                        elif slope>0.:low=max(low,-value/slope)
                        else:high=min(high,-value/slope)
                        if high-low<=min_fraction:
                            valid=False;break
                    if not valid:break
                if valid and high-low>min_fraction:
                    intervals[number,0]=low;intervals[number,1]=high;number+=1
        if not number:continue
        order=np.argsort(intervals[:number,0])
        low=intervals[order[0],0];high=intervals[order[0],1]
        for k in range(1,number+1):
            if k<number and intervals[order[k],0]<=high+min_fraction:
                high=max(high,intervals[order[k],1]);continue
            indices[count]=j
            output[count,0]=segments[j,0] if low==0. else start+low*delta
            output[count,1]=segments[j,1] if high==1. else start+high*delta
            count+=1
            if k<number:low=intervals[order[k],0];high=intervals[order[k],1]
    return indices[:count],output[:count]


class BoundaryDomains:
    """Call-local packing/cache; CPU domains are immutable published metadata."""
    def __init__(self):
        self.records={}

    def record(self, domain):
        found=self.records.get(id(domain))
        if found is not None:return found[1:]
        branches=(np.empty((0,4)),) if domain is None else domain
        lengths=np.asarray([len(branch) for branch in branches],dtype=np.int64)
        planes=np.concatenate(branches).astype(np.float64,copy=False) if len(branches) else np.empty((0,4))
        signature=(b'\0' if domain is None else b'\1')+np.asarray([len(branches)],np.int64).tobytes()+lengths.tobytes()+planes.tobytes()
        record=(domain,lengths,planes,signature)
        self.records[id(domain)]=record
        return record[1:]

    def key(self, domain):
        return self.record(domain)[2]

    def trim(self, segments, own, peers):
        if own is None and all(domain is None for domain in peers):
            return np.arange(len(segments),dtype=np.int64),segments
        records=[self.record(domain) for domain in [own,*peers]]
        domain_offsets=np.r_[0,np.cumsum([len(record[0]) for record in records])].astype(np.int64)
        lengths=np.concatenate([record[0] for record in records])
        plane_offsets=np.r_[0,np.cumsum(lengths)].astype(np.int64)
        planes=np.concatenate([record[1] for record in records])
        function=_trim_segments if enabled() else getattr(_trim_segments,'py_func',_trim_segments)
        return function(segments,domain_offsets,plane_offsets,planes)

"""GaussianCutSim core / spatial_index implementation."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace, is_dataclass
import hashlib
import numpy as np
from cutsim.core.sweep import BallSweep, FlatSweep


class _SweepIndexNode:
    """Immutable spatial hierarchy node; old roots remain valid for playback."""
    __slots__ = ('lo','hi','minimum','left','right','index','height','last','small',
                 'small_lo','small_hi','small_original_hi')

    def __init__(self, lo=None, hi=None, minimum=0., index=-1, left=None, right=None, original_hi=None):
        self.left=left; self.right=right; self.index=index
        if left is None:
            self.lo=lo; self.hi=hi; self.minimum=minimum
            self.height=0; self.last=index
            self.small=np.array([index],dtype=np.int64)
            self.small_lo=lo[None,:]; self.small_hi=hi[None,:]
            self.small_original_hi=(hi if original_hi is None else original_hi)[None,:]
        else:
            self.lo=np.minimum(left.lo,right.lo); self.hi=np.maximum(left.hi,right.hi)
            self.minimum=min(left.minimum,right.minimum)
            self.height=1+max(left.height,right.height)
            self.last=max(left.last,right.last)
            if (left.small is not None and right.small is not None
                    and len(left.small)+len(right.small)<=6):
                self.small=np.concatenate((left.small,right.small))
                self.small_lo=np.concatenate((left.small_lo,right.small_lo))
                self.small_hi=np.concatenate((left.small_hi,right.small_hi))
                self.small_original_hi=np.concatenate((left.small_original_hi,right.small_original_hi))
            else:
                self.small=None; self.small_lo=self.small_hi=self.small_original_hi=None

    @staticmethod
    def pair(left,right):
        return _SweepIndexNode(left=left,right=right)


def _sweep_tree_balance(left,right):
    pair=_SweepIndexNode.pair
    if left.height>right.height+1:
        a,b=left.left,left.right
        if a.height>=b.height:
            return pair(a,pair(b,right))
        return pair(pair(a,b.left),pair(b.right,right))
    if right.height>left.height+1:
        a,b=right.left,right.right
        if b.height>=a.height:
            return pair(pair(left,a),b)
        return pair(pair(left,a.left),pair(a.right,b))
    return pair(left,right)


def _sweep_tree_insert(root,leaf):
    if root is None: return leaf
    if root.left is None: return _SweepIndexNode.pair(root,leaf)
    def cost(node):
        old=node.hi-node.lo
        new=np.maximum(node.hi,leaf.hi)-np.minimum(node.lo,leaf.lo)
        # Surface area works for thin or planar boxes as well as volumes.
        return float(new[0]*new[1]+new[1]*new[2]+new[2]*new[0]
                     -old[0]*old[1]-old[1]*old[2]-old[2]*old[0])
    a,b=root.left,root.right
    if (cost(a),a.height)<=(cost(b),b.height):
        return _sweep_tree_balance(_sweep_tree_insert(a,leaf),b)
    return _sweep_tree_balance(a,_sweep_tree_insert(b,leaf))


class _HistoryPrefix:
    """Constant-time prefix view, with a private append-only tail for previews."""
    def __init__(self, source, count):
        self.source=source; self.count=count; self.tail=[]

    def __len__(self): return self.count+len(self.tail)

    def __getitem__(self,index):
        if isinstance(index,slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index<0: index+=len(self)
        if not 0<=index<len(self): raise IndexError(index)
        return self.source[index] if index<self.count else self.tail[index-self.count]

    def __iter__(self):
        for i in range(self.count): yield self.source[i]
        yield from self.tail

    def append(self,value): self.tail.append(value)


class SweepSpatialIndex:
    """Persistent balanced AABB tree of registered analytic cutter sweeps.

    Additions update only the insertion path. Queries descend intersecting
    branches, and history prefixes reuse previous immutable roots. ``lo/hi``
    retain the original broad-phase boxes for compatibility; tree bounds also
    enclose the complete ball capsule when its shank is unusually short.
    Registered sweep geometry must not be edited in place after insertion.
    """
    def __init__(self):
        self.lo=[]; self.hi=[]; self.min_sdf=[]; self._roots=[]
        self.tokens=[]; self.shapes=[]; self.scales=[]; self.supported=[]; self.digests=[]
        self.last_query_nodes=0; self.last_value_nodes=0; self.last_value_pairs=0
        self.query_nodes=0; self.query_calls=0

    def add(self,sweep):
        lo,hi=sweep.world_aabb()
        lo=np.asarray(lo,dtype=np.float64).copy(); hi=np.asarray(hi,dtype=np.float64).copy()
        self.lo.append(lo); self.hi.append(hi); self.min_sdf.append(-float(sweep.radius))
        bounds_hi=hi.copy()
        if type(sweep) is BallSweep:
            bounds_hi[2]=max(bounds_hi[2],sweep.center_z+sweep.radius)
        leaf=_SweepIndexNode(lo,bounds_hi,-float(sweep.radius),len(self.lo)-1,original_hi=hi)
        self._roots.append(_sweep_tree_insert(self._roots[-1] if self._roots else None,leaf))
        supported=type(sweep) in (FlatSweep,BallSweep)
        self.supported.append(supported and (self.supported[-1] if self.supported else True))
        token=hashlib.blake2b(digest_size=24)
        token.update(type(sweep).__name__.encode())
        if supported or is_dataclass(sweep):
            for field in fields(sweep):
                value=getattr(sweep,field.name)
                token.update(value.tobytes() if isinstance(value,np.ndarray) else repr(value).encode())
                token.update(b'\0')
        if supported:
            shape=tuple(map(float,(*sweep.a_xy,*sweep.b_xy,sweep.radius,
                        sweep.center_z if type(sweep) is BallSweep else sweep.z_lo,sweep.z_hi)))+(type(sweep) is BallSweep,)
        else:
            shape=(0.,)*7+(False,)
        self.tokens.append(token.digest()); self.shapes.append(shape)
        self.scales.append(max(max(abs(x) for x in shape[:7]),self.scales[-1] if self.scales else 0.))
        previous=self.digests[-1] if self.digests else b''
        self.digests.append(hashlib.blake2b(previous+token.digest(),digest_size=24).digest())

    def prefix(self,count):
        count=max(0,min(int(count),len(self.lo)))
        out=SweepSpatialIndex()
        for name in ('lo','hi','min_sdf','_roots','tokens','shapes','scales','supported','digests'):
            setattr(out,name,_HistoryPrefix(getattr(self,name),count))
        return out

    @staticmethod
    def _bound(p,lo,hi,minimum):
        delta=np.maximum(np.maximum(lo-p,p-hi),0.)
        outside=np.any(delta>0.,axis=1)
        return np.where(outside,np.linalg.norm(delta,axis=1),minimum)

    def lower_bound(self,points,index):
        return self._bound(np.asarray(points,dtype=np.float64).reshape(-1,3),
                           self.lo[index],self.hi[index],self.min_sdf[index])

    def query_aabb(self,lo,hi,*,upto=None,original_boxes=True):
        limit=len(self.lo) if upto is None else max(0,min(int(upto),len(self.lo)))
        self.last_query_nodes=0
        self.query_calls+=1
        if not limit: return []
        result=[]; stack=[self._roots[limit-1]]
        while stack:
            node=stack.pop(); self.last_query_nodes+=1
            if node.small is not None:
                boxes_hi=node.small_original_hi if original_boxes else node.small_hi
                hit=np.all(boxes_hi>=lo,axis=1)&np.all(node.small_lo<=hi,axis=1)
                result.extend(node.small[hit].tolist())
                continue
            if np.any(node.hi<lo) or np.any(node.lo>hi): continue
            if node.left is None:
                i=node.index
                if original_boxes and (np.any(self.hi[i]<lo) or np.any(self.lo[i]>hi)): continue
                result.append(i)
            else:
                stack.extend((node.left,node.right))
        # Index order determines distance ties and compositing order.
        result.sort()
        self.query_nodes+=self.last_query_nodes
        return result

    def nearest_surface_tools(self,p,best,owners,sweeps):
        """Exact unsigned nearest surface, preserving earliest-history ties."""
        if not len(self.lo) or not len(p): return owners
        ordinal=np.full(len(p),-1,dtype=np.int64)
        stack=[(self._roots[-1],np.arange(len(p)))]
        while stack:
            node,ids=stack.pop()
            # Inside a containing box the unsigned lower bound is zero.
            delta=np.maximum(np.maximum(node.lo-p[ids],p[ids]-node.hi),0.)
            bound=np.linalg.norm(delta,axis=1)
            eps=64*np.finfo(float).eps*np.maximum(1.,np.maximum(bound,best[ids]))
            ids=ids[bound<=best[ids]+eps]
            if not len(ids): continue
            if node.left is None:
                i=node.index; d=np.abs(sweeps[i].sdf(p[ids]))
                take=(d<best[ids])|((d==best[ids])&(i<ordinal[ids]))
                ii=ids[take]; best[ii]=d[take]; ordinal[ii]=i; owners[ii]=int(sweeps[i].tool_id)
            else:
                stack.extend(((node.right,ids),(node.left,ids)))
        return owners

    def material_value(self,p,stock,sweeps,limit):
        self.last_value_nodes=0; self.last_value_pairs=0
        if not limit or not len(p): return stock,0
        # Only distances below -stock can alter max(stock, -min(sweep_sdf)).
        # This finite initial bound avoids finding an irrelevant distant cutter.
        cut=-stock.copy(); evaluated=0
        stack=[(self._roots[limit-1],np.arange(len(p)))]
        while stack:
            node,ids=stack.pop()
            self.last_value_nodes+=1; self.last_value_pairs+=len(ids)
            bound=self._bound(p[ids],node.lo,node.hi,node.minimum)
            # Conservative roundoff guard, especially at touching surfaces.
            tolerance=64*np.finfo(float).eps*np.maximum(1.,np.maximum(np.abs(bound),np.abs(cut[ids])))
            ids=ids[bound<=cut[ids]+tolerance]
            if not len(ids): continue
            if node.small is not None:
                # Fixed-size leaves amortize Python traversal/mask overhead.
                # Six is a constant, independent of the total history length.
                pp=p[ids]; best=cut[ids]
                for i in node.small:
                    d=sweeps[i].sdf(pp); evaluated+=len(ids)
                    best=np.minimum(best,d)
                cut[ids]=best
            else:
                a,b=node.left,node.right
                # Newer cuts commonly establish a tighter bound first.
                if a.last>b.last: a,b=b,a
                stack.extend(((a,ids),(b,ids)))
        return np.maximum(stock,-cut),evaluated

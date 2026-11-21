"""GPU-resident current-GS -> profile Dexel path.

Input snapshot packing/upload is measured separately. GPU builds a Morton BVH,
searches finite clipped GS, resolves material signs, sorts variable-length hits,
and assembles intervals. Only bounded allocation/statistics scalars are read
back before cutting. The NumPy wrappers serve existing clipping/debug consumers;
they are not called by seed_device or endpoint_mask_device.
"""
from time import perf_counter
import numpy as np
from cutsim.rendering.gs_extent import support_factor, support_radius
from cutsim.cuda.current_gs_surface import CurrentGSSurface,_CUDA
from cutsim.cuda.current_gs_resident_kernels import RESIDENT_CUDA
from cutsim.cuda.profile_ray_domain import ray_bounds




class ResidentCurrentGSSurface(CurrentGSSurface):
    def __init__(self,snapshot,backend='cuda',*,cache=None,max_hits=2_000_000,selection=None):
        if backend!='cuda':raise ValueError('Resident GS pipeline requires CUDA; no CPU fallback')
        if not isinstance(max_hits,int) or isinstance(max_hits,bool) or max_hits<1:raise ValueError('Positive hit budget required')
        started=perf_counter()
        # Pack the current CPU publication once; no CPU spatial index or
        # per-ray/per-query candidate lists are built.
        self.sweep_selection=selection
        super().__init__(snapshot,backend,cache=None,spatial_index=False,selection=selection)
        packed=perf_counter()
        self.pack_upload_ms=(packed-started)*1000
        self.max_hits=max_hits;self.readback_bytes=0;self.control_readback_bytes=0
        self.upload_bytes=self.geometry.nbytes;self.leaves=1<<max(0,(max(1,len(self.rows))-1).bit_length())
        if 2*self.leaves*6*8>128*1024*1024:raise ValueError('Resident BVH size budget exceeded')
        rt=self.rt;cp=rt.cp
        with rt.context():
            module=cp.RawModule(code=(_CUDA+RESIDENT_CUDA).replace("exp(-2*",f"exp(-{.5*support_factor(snapshot)**2}*"),options=('--fmad=false',))
            names=('codes','leaves','parents','query','segments','enclosure','boundary_risk','hits','intervals')
            self.resident_kernels={name:module.get_function('resident_'+name) for name in names}
            self.device_rows=self._upload(self.rows)
            self._adaptive_identity=None
            self.boxes=cp.empty((2*self.leaves,6),cp.float64)
            self.counters=cp.zeros(11,cp.int64)
            codes=cp.empty(len(self.rows),cp.uint32)
            if len(self.rows):
                centers=self.device_geometry[:,:3]
                self._launch('codes',len(self.rows),(self.device_geometry,np.int64(len(self.rows)),
                    cp.min(centers,axis=0),cp.max(centers,axis=0),codes))
            self.order=cp.argsort(codes).astype(cp.int64)
            self._launch('leaves',self.leaves,(self.device_geometry,self.order,np.int64(len(self.rows)),np.int64(self.leaves),self.boxes))
            level=self.leaves//2
            while level:
                self._launch('parents',level,(self.boxes,np.int64(level),np.int64(level)));level//=2
            rt.stream.synchronize()
        self.build_ms=(perf_counter()-packed)*1000

    def _launch(self,name,count,args):
        if count:self.resident_kernels[name](((count+127)//128,),(128,),args)

    def _upload(self,array):
        self.upload_bytes+=array.nbytes
        return self.rt.cp.asarray(array)

    def _download(self,array,*,control=False):
        result=self.rt.cp.asnumpy(array,stream=self.rt.stream)
        if control:self.control_readback_bytes+=result.nbytes
        else:self.readback_bytes+=result.nbytes
        return result

    def _tree_args(self):
        return (self.device_geometry,self.boxes,self.order,self.device_rows,np.int64(len(self.rows)),np.int64(self.leaves))

    def pull_stats(self):
        with self.rt.context():values=self._download(self.counters,control=True)
        keys=('query_points','query_pairs','ray_pairs','material_edge_queries','material_edge_resolved',
              'material_side_changes','material_confirm_queries','material_conflicts',
              'endpoint_edge_queries','endpoint_edge_resolved','endpoint_side_changes')
        self.stats.update(zip(keys,map(int,values)))

    def query_device(self,points,*,material=False,endpoint=False):
        cp=self.rt.cp
        values=cp.empty(len(points),cp.float64);owners=cp.empty(len(points),cp.int64)
        self._launch('query',len(points),(points,np.int64(len(points)),*self._tree_args(),
            np.int32(2 if endpoint else int(material)),values,owners,self.counters))
        return values,owners

    def query(self,points,*,use_cache=True):
        points=np.ascontiguousarray(points,dtype=float).reshape(-1,3)
        with self.rt.context():
            values,owners=self.query_device(self._upload(points))
            return self._download(values),self._download(owners)

    def enclosure_device(self,points):
        """Current finite GS evidence: 1=opposed exits, -1=air/on face, 0=unknown.

        GPU-only, bounded to three opposed axis pairs; no cutter history or
        nearest-plane extrapolation. Intended for addition-only completion.
        """
        points=self.rt.cp.ascontiguousarray(points,dtype=self.rt.cp.float64)
        evidence=self.rt.cp.empty(len(points),self.rt.cp.int8)
        self._launch('enclosure',len(points),(points,np.int64(len(points)),*self._tree_args(),evidence,self.counters))
        return evidence

    def material_query(self,points,*,use_cache=True,stats_prefix='material'):
        points=np.ascontiguousarray(points,dtype=float).reshape(-1,3)
        with self.rt.context():
            values,owners=self.query_device(self._upload(points),material=True,endpoint=stats_prefix=='endpoint')
            return self._download(values),self._download(owners)

    def segment_boundaries(self,inside,outside):
        inside=np.ascontiguousarray(inside,dtype=float).reshape(-1,3);outside=np.ascontiguousarray(outside,dtype=float).reshape(-1,3)
        if len(inside)!=len(outside):raise ValueError('Matching segment endpoint counts required')
        with self.rt.context():
            owners=self.rt.cp.empty(len(inside),self.rt.cp.int64)
            self._launch('segments',len(inside),(self._upload(inside),self._upload(outside),np.int64(len(inside)),
                *self._tree_args(),owners,self.counters))
            return self._download(owners)

    def boundary_risk_device(self,snapshot):
        """GPU boundary probes for Adaptive, before any new GS clipping."""
        cp=self.rt.cp;count=len(snapshot.endpoints['positions'])
        if self._adaptive_identity is None:
            self._adaptive_identity=self._upload(np.ascontiguousarray(self.snapshot.material_surface_ids[self.rows],dtype=np.int64))
        proposed=self._upload(np.ascontiguousarray(np.column_stack((snapshot.endpoints['positions'],
            snapshot.gaussian.u,snapshot.gaussian.v,snapshot.endpoints['normals'],
            2*snapshot.gaussian.attributes['scale'][:,0])),dtype=float))
        ids=self._upload(np.ascontiguousarray(snapshot.material_surface_ids,dtype=np.int64))
        risk=cp.empty(count,cp.uint8)
        self._launch('boundary_risk',count,(proposed,ids,np.int64(count),*self._tree_args(),self._adaptive_identity,risk,self.counters))
        return risk

    def hits_device(self,origins,directions,domain):
        cp=self.rt.cp;n=len(origins);counts=cp.empty(n,cp.int64);dummy=cp.empty(0,cp.float64)
        self._launch('hits',n,(origins,directions,domain,np.int64(n),*self._tree_args(),counts,counts,dummy,np.int32(0),self.counters))
        offsets=cp.concatenate((cp.zeros(1,cp.int64),cp.cumsum(counts)))
        total=int(self._download(offsets[-1:],control=True)[0])
        if total>self.max_hits:raise ValueError('Resident GS hit budget exceeded; no hits truncated')
        hits=cp.empty((total,5),cp.float64)
        self._launch('hits',n,(origins,directions,domain,np.int64(n),*self._tree_args(),offsets,counts,hits,np.int32(1),self.counters))
        if total:
            order=cp.lexsort(cp.stack((hits[:,3],hits[:,1],hits[:,0])))
            hits=cp.ascontiguousarray(hits[order])
        return hits,counts,offsets

    def intersections(self,grid):
        with self.rt.context():
            o,d,b=[self._upload(np.ascontiguousarray(a,dtype=float)) for a in (grid.origins,grid.directions,ray_bounds(grid))]
            hits,_,_=self.hits_device(o,d,b)
            return self._download(hits)

    def endpoint_mask_device(self,points,ray_has_hit):
        cp=self.rt.cp;values,owners=self.query_device(points,material=True,endpoint=True)
        valid=owners>=0;index=cp.searchsorted(self.device_rows,cp.maximum(owners,0))
        if not len(self.rows):return cp.zeros(len(points),cp.bool_)
        g=self.device_geometry[cp.minimum(index,len(self.rows)-1)]
        delta=points-g[:,:3]
        height=(delta[:,0]*g[:,9]+delta[:,1]*g[:,10])+delta[:,2]*g[:,11]
        return valid&(values < -1e-8)&((height<0)|ray_has_hit)


    def close(self):
        self._analytic_ray_evidence=None
        self.residual_evaluation=None
        self.residual_samples=None
        self.device_rows=self.boxes=self.order=self.counters=self._adaptive_identity=None
        super().close()

"""Bounded parallel native subdivision with ordered, single-threaded publishing.

Workers own their complete parent calculation. They never modify clouds,
render attributes, caches or shared diagnostic dictionaries. Boundary scoring,
pruning and Float32 encoding retain their original order inside each parent.
"""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os
import numpy as np


@lru_cache(maxsize=None)
def _executor(workers):
    return ThreadPoolExecutor(max_workers=workers,thread_name_prefix='GS-subdivision',initializer=_initialize_worker)


def _initialize_worker():
    from cutsim.rendering.fallback_kernels import parent_worker
    parent_worker.single_thread=True


def _parent_job(record,regions,finite,minimum,stock=None):
    from cutsim.rendering.clip_fallback import build_pending_fallback
    limit=2048
    while True:
        result=build_pending_fallback(record,regions,prune=True,
            base_depth=0 if finite else 3,max_children=limit,
            preserve_simple_regions=finite,min_child_sigma=minimum,stock_planes=stock)
        if not finite or 'budget' not in result['incomplete_reasons']:return result
        limit*=2


def resolve_parallel(out,metadata):
    """Return False to use the serial resolver for small/disabled jobs."""
    from cutsim.rendering import fallback_kernels as kernels
    from cutsim.geometry.strip_kernels import configure_threads
    if os.environ.get('CUTSIM_PARENT_PARALLEL','1')=='0' or not kernels.enabled():return False
    rows=np.flatnonzero(out['clip_pending']&~out['clip_empty'])
    rows=[int(i) for i in rows if 'pending_fallback' not in metadata.get(int(i),{})
          and metadata.get(int(i),{}).get('fallback_regions_local') is not None]
    if len(rows)<8:return False
    workers=configure_threads()
    if workers<2:return False
    from cutsim.geometry.sampling_density import DensityPolicy
    from cutsim.rendering.exact_draw_cache import fallback_cache,array_key
    from cutsim.rendering.gs_schema import GS_FIELDS
    from cutsim.rendering.coverage_cuda import backend_key, evaluator
    coverage_policy = backend_key()
    policy=DensityPolicy(getattr(out,'gs_density',1.))
    pool=_executor(workers)
    # At most two native jobs per worker are queued/retained; never materialize
    # subdivisions for all parents at once. Results are committed by row order.
    batch_size=2*workers
    for start in range(0,len(rows),batch_size):
        jobs={};ordered=[]
        try:
            for i in rows[start:start+batch_size]:
                diagnostic=metadata[i];regions=diagnostic['fallback_regions_local']
                record={name:value[i] for name,value in out.items()}
                finite=diagnostic.get('finite_strip_domain',False)
                stock=diagnostic.get('stock_planes_local')
                minimum=policy.draw_floor(regions)*getattr(out,'draw_subdivision_floor_scale',1.)
                key=(kernels.subdivision_policy(),coverage_policy,finite,minimum,tuple((name,array_key(record[name]))
                    for name in GS_FIELDS+('coverage_logit',) if name in record),
                    tuple(array_key(region) for region in regions),None if stock is None else array_key(stock))
                cached=fallback_cache.get(key)
                ordered.append((diagnostic,key,cached))
                if cached is not None or key in jobs:continue
                evaluator()  # Initialize CUDA on the caller before launching workers.
                jobs[key]=pool.submit(_parent_job,record,regions,finite,minimum,stock)
            completed={}
            for diagnostic,key,cached in ordered:
                if cached is None:
                    if key not in completed:
                        cached=jobs[key].result()
                        fallback_cache.put(key,cached)
                        completed[key]=cached
                    cached=completed[key]
                diagnostic['pending_fallback']=cached
        except BaseException:
            # Running native jobs only hold private memory. Observe their end
            # before propagating failure; never leave writes racing a reset.
            for job in jobs.values():job.cancel()
            for job in jobs.values():
                try:job.result()
                except BaseException:pass
            raise
    return True

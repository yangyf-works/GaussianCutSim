"""Publish grouped analytic clipping for current Tool-profile GS."""
import numpy as np

def update_current_gs_clipping(snapshot,previous,kept,surface,sweep,backend,*,new_only=False,stock_bounds=None,current_surface_id=None,certified_fragments=0,certified_old=None,prepared_old=None,collect_edges=False,gpu_prepare=False,gs_pairs=False,projection=None,analytic_old=False,sweep_workspace=None,analytic_pairs=False,analytic_sampled=False):
    if backend!='cuda':raise ValueError('Grouped analytic clipping requires CUDA')
    from cutsim.rendering.sample_priority_clip import update_sample_priority_clipping
    cached=getattr(surface,'residual_samples',None)
    new_samples=None
    if cached is not None:
        from cutsim.rendering.analytic_residual_retirement import sample_residual
        nold=int(np.count_nonzero(kept));count=len(snapshot.endpoints['positions'])-nold
        edge=(sweep_workspace or {}).get('new_edge_mask')
        if edge is None:edge=np.ones(count,bool) # standalone/repair callers
        if np.shape(edge)!=(count,):raise ValueError('New edge mask must match generated GS')
        new_samples,_,_=sample_residual(surface,snapshot,nold+np.flatnonzero(edge),sweep,
                                        stock_bounds,current_cut=False)
    return update_sample_priority_clipping(snapshot,previous,kept,collect_edges=collect_edges,gpu_prepare=gpu_prepare,
        preserve_nonempty=bool(sweep_workspace and sweep_workspace.get('preserve_nonempty')),
        retained_sweep=sweep if sweep_workspace and sweep_workspace.get('retained_region') else None,
        current_surface_ids=current_surface_id,stock_bounds=stock_bounds,
        residual_samples=cached,new_samples=new_samples)

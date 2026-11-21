"""Batch only the quadrature of terminal Top-2 decisions, with exact ordering.

Keep the existing feasible-line solver, canonical ties, 512 decision samples
inside 2 sigma and NumPy reduction order. No approximation or wider search.
"""
import numpy as np


def top_two_many(groups, mandatory):
    from cutsim.rendering.clipping import _sweep_line_interval,_sweep_decision_quadrature
    result=[None]*len(groups)
    buckets={}
    for i,q in enumerate(groups):
        if len(q)<=2:result[i]=(q.copy(),list(range(len(q))))
        else:buckets.setdefault(len(q),[]).append(i)
    points,weights=_sweep_decision_quadrature()
    for count,indices in buckets.items():
        # Keep temporaries bounded even for complex domains. Independent
        # parents already run in a bounded pool; this is per-parent batching.
        batch=max(1,min(64,262144//(len(points)*count)))
        for start in range(0,len(indices),batch):
            rows=indices[start:start+batch]
            q=np.stack([groups[i] for i in rows])
            from cutsim.rendering.domain_batch import enabled,line_peak_scores
            native=enabled()
            logs=line_peak_scores(q) if native else np.full((len(rows),count),-np.inf)
            canonical=[];first=[]
            for k,i in enumerate(rows):
                if not native:
                    for j,p in enumerate(q[k]):
                        interval=_sweep_line_interval(q[k],j)
                        if interval is not None:
                            t=np.clip(0.,*interval)
                            logs[k,j]=-.5*(p[2]**2+t*t)
                order=sorted(range(count),key=lambda j:tuple(q[k,j]))
                canonical.append(order)
                first.append(int(mandatory[i]) if mandatory[i] is not None else max(order,key=lambda j:logs[k,j]))
            cut=points@q[:,:,:2].transpose(0,2,1)+q[:,None,:,2]<0.
            excluded=cut[np.arange(len(rows)),:,first]
            marginal=np.sum(weights[None,:,None]*(cut&~excluded[:,:,None]),axis=1)/weights.sum()
            scores=logs+np.log(np.maximum(marginal,1e-12))
            for k,i in enumerate(rows):
                second=max((j for j in canonical[k] if j!=first[k]),key=lambda j:scores[k,j])
                chosen=[first[k],second]
                result[i]=(q[k,chosen].copy(),chosen)
    return result


def terminal_many(leaves, stock, origin, inverse, sigma):
    """Same mandatory stock constraints and child order as the scalar path."""
    groups=[];mandatory=[];ready={}
    for i,(position,child_sigma,candidates) in enumerate(leaves):
        q=np.asarray(candidates);required=[]
        if stock is not None and len(stock):
            center=inverse@(position.astype(float)-origin)
            ratio=float(child_sigma)/sigma
            hard=np.column_stack((stock[:,:2],(stock[:,:2]@center+stock[:,2])/ratio))
            for p in hard[hard[:,2]<3.+1e-7]:
                match=np.flatnonzero(np.all(np.abs(q-p)<=1e-8*np.maximum(1.,np.abs(p)),axis=1))
                if len(match):index=int(match[0])
                else:index=len(q);q=np.vstack((q,p))
                if index not in required:required.append(index)
        if len(required)>2:
            raise ValueError('Terminal GS intersects more than two mandatory stock planes; finer subdivision is required')
        if len(required)==2:ready[i]=(q[required].copy(),required,q)
        groups.append(q);mandatory.append(required[0] if len(required)==1 else None)
    pending=[i for i in range(len(groups)) if i not in ready]
    chosen=top_two_many([groups[i] for i in pending],[mandatory[i] for i in pending])
    ready.update((i,(planes,indices,groups[i])) for i,(planes,indices) in zip(pending,chosen))
    return [ready[i] for i in range(len(groups))]

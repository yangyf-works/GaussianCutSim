"""GaussianCutSim core / scipy_support implementation."""
from __future__ import annotations




try:
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as sparse_connected_components
    _HAVE_SCIPY_GRAPH = True
except Exception:
    cKDTree = coo_matrix = sparse_connected_components = None
    _HAVE_SCIPY_GRAPH = False

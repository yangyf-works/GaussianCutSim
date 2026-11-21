"""Generator-owned tool witnesses and addition-only adaptive Dexels.

The finalization stage requests rows; this module owns chart evaluation,
material evidence, ray traversal and GS fitting. It never applies retirement.
"""
from dataclasses import dataclass, field
from types import SimpleNamespace
from time import perf_counter
import numpy as np
from cutsim.cuda.adaptive_profile import CHARTS, FIELDS, _generate
from cutsim.rendering.coverage_audit import CoverageProbes


@dataclass(frozen=True)
class ToolCoverageGrid:
    grid: object
    parents: np.ndarray
    probes: CoverageProbes
    unchecked: dict
    block: int = 0
    before: object = None
    material_stats: dict = field(default_factory=dict)

    def request(self, rows, max_rays):
        if not isinstance(max_rays, int) or isinstance(max_rays, bool) or max_rays < 0:
            raise ValueError('Coverage ray budget must be a nonnegative integer')
        rows = np.unique(np.asarray(rows, dtype=np.int64))
        if np.any(rows<0) or np.any(rows>=len(self.parents)):
            raise ValueError('Coverage requests must reference the tool witness grid')
        if len(rows) and not self.probes.expected[rows].any(axis=1).all():
            raise ValueError('Coverage requests require material evidence')
        return rows[:max_rays], max(0, len(rows)-max_rays)


def coverage_material_evidence(surface, points, directions):
    """Stage 5 naming adapter for the shared finite material predicate."""
    from cutsim.cuda.material_evidence import finite_material_evidence
    states, stats = finite_material_evidence(surface, points, directions)
    return states, {'coverage_' + key: value for key, value in stats.items()}


def select_coverage_cells(charts, max_probes, divisions, boundary_cells=None):
    """Union the ordinary grid with the separately bounded boundary grid."""
    from cutsim.cuda.coverage_candidates import BOUNDARY_MAX_PROBES
    supported = np.flatnonzero(np.isin(charts, CHARTS))
    capacity = max_probes//(divisions*divisions)
    ordinary = supported
    if len(ordinary) > capacity:
        ordinary = ordinary[np.linspace(0, len(ordinary)-1, capacity, dtype=np.int64)]
    boundary = np.empty(0, np.int64) if boundary_cells is None else np.asarray(boundary_cells, np.int64)
    if np.any(boundary < 0) or np.any(boundary >= len(charts)):
        raise ValueError('Boundary cells must reference the base tool grid')
    boundary = np.intersect1d(boundary, supported)
    boundary_capacity = BOUNDARY_MAX_PROBES//(divisions*divisions)
    if len(boundary) > boundary_capacity:
        boundary = boundary[np.linspace(0, len(boundary)-1, boundary_capacity, dtype=np.int64)]
    parents = np.union1d(ordinary, boundary)
    unchecked = dict(tool_cells=len(supported)-len(parents),
                     unsupported_cells=len(charts)-len(supported), material_points=0)
    selection = dict(coverage_ordinary_cells=len(ordinary), coverage_boundary_cells=len(boundary),
                     coverage_duplicate_cells=len(ordinary)+len(boundary)-len(parents),
                     coverage_selected_cells=len(parents), coverage_unified=boundary_cells is not None)
    return parents, unchecked, selection


def prepare_tool_coverage(surface, atlas, command, sweep, block, stock_bounds,
                          max_probes=4096, divisions=4, *, boundary_cells=None):
    """Sample all supported tool cells, including those with zero generated GS."""
    template = getattr(atlas, 'profile_template', atlas)
    parents, unchecked, selection = select_coverage_cells(
        template.charts, max_probes, divisions, boundary_cells)
    material_stats = dict(coverage_material_method='finite_analytic_faces',
        coverage_analytic_faces=0, coverage_analytic_material_points=0,
        coverage_analytic_air_points=0, coverage_analytic_unknown_points=0,
        coverage_analytic_cut_rejections=0)
    rt = surface.rt; cp = rt.cp
    with rt.context():
        model = SimpleNamespace(_origins=cp.asarray(template.origins), _directions=cp.asarray(template.directions))
        grid, source, _ = _generate(surface, model, template, parents, None, None, command, block,
            len(parents)*divisions*divisions, risk_mask=np.ones(len(parents), bool), divisions=divisions)
        if grid is None:
            cpu = {name: np.asarray(getattr(template, name))[:0].copy() for name in FIELDS}
            cpu['charts'] = np.empty(0, np.int64)
            points = normals = np.empty((0, 3)); source = np.empty(0, np.int64)
            expected = np.empty(0, bool)
        else:
            q = grid.origins+grid.surface_distance[:, None]*grid.directions
            valid = (grid.surface_distance>=grid.ray_bounds[:, 0])&(grid.surface_distance<=grid.ray_bounds[:, 1])
            valid &= cp.all((q>=cp.asarray(stock_bounds[0]))&(q<=cp.asarray(stock_bounds[1])), axis=1)
            rows = cp.flatnonzero(valid)
            points = surface._download(q[rows]); normals = surface._download(-grid.directions[rows])
            envelope = np.flatnonzero(np.abs(sweep.sdf(points))<=1e-7)
            rows = rows[cp.asarray(envelope)]
            points = points[envelope]; normals = normals[envelope]
            cpu = {name: surface._download(getattr(grid, name)[rows]) for name in FIELDS}
            source = surface._download(source[rows])
            evidence, material_stats = coverage_material_evidence(surface, points, -normals)
            expected = evidence == 1
            unchecked['material_points'] = material_stats['coverage_analytic_unknown_points']
    probes = CoverageProbes(points, normals, np.repeat(expected[:, None], 2, axis=1),
        np.full(len(points), 1e-7), np.full(len(points), -1, np.int64),
        np.arange(len(points), dtype=np.int64), float(command.radius))
    material_stats.update(selection)
    return ToolCoverageGrid(SimpleNamespace(**cpu), source, probes, unchecked,
                            block, surface.snapshot, material_stats)


def generate_coverage_gs(surface, witnesses, rows, command, sweep, block, group,
                         material_group, box, ray_offset, ray_margin=None):
    """Generate on the current tool surface using finite analytic evidence.

    BEFORE owns the material equations, not the new surface geometry. This
    addition-only path performs no old-GS hit traversal, retirement calculation
    or planar completion. Every emitted centre must pass the same analytic
    material and already-cut rejection used by the coverage witnesses.
    """
    if witnesses.block != block or witnesses.before is not surface.snapshot:
        raise ValueError('Coverage repair request belongs to a different block or BEFORE snapshot')
    rows, _ = witnesses.request(rows, len(rows))
    raw = SimpleNamespace(**{name: getattr(witnesses.grid, name)[rows] for name in FIELDS})
    raw.charts = CHARTS[raw.charts]
    new, descriptor, stats = generate_tool_gs(surface, raw, command, sweep, block, group,
        material_group, box, ray_offset, ray_margin, material_predicate=coverage_material_evidence)
    return new, descriptor, witnesses.parents[rows], stats


def generate_tool_gs(surface, raw, command, sweep, block, group, material_group,
                     box, ray_offset, ray_margin=None, *, stock_bounds=None, material_predicate=None):
    """Shared addition-only generation for cell subdivision and coverage repair.

    Every centre is rechecked against the BEFORE finite material evidence.
    No old-GS hit mapping, deletion, or coverage audit runs here.
    """
    from cutsim.cuda.center_ray_update import center_grid
    from cutsim.cuda.profile_program import _snapshot
    from cutsim.cutting.tool_profile_state import RAY_STRIDE
    from cutsim.cuda.material_evidence import finite_material_evidence
    if ray_offset < 0 or ray_offset+len(raw.origins)>RAY_STRIDE:
        raise ValueError('Additional rays exceed the block ray-ID range')
    started = perf_counter()
    grid = center_grid(raw, sweep, box, margin=ray_margin)
    points = grid.origins + grid.surface_distance[:, None] * grid.directions
    valid = ((grid.surface_distance >= grid.ray_bounds[:, 0]) &
             (grid.surface_distance <= grid.ray_bounds[:, 1]))
    valid &= np.abs(sweep.sdf(points)) <= 1e-7
    if stock_bounds is not None:
        valid &= np.all((points>=stock_bounds[0])&(points<=stock_bounds[1]), axis=1)
    candidates = np.flatnonzero(valid)
    ready = perf_counter()
    predicate = finite_material_evidence if material_predicate is None else material_predicate
    evidence, _ = predicate(surface, points[candidates], grid.directions[candidates])
    selected = candidates[evidence == 1]
    evaluated = perf_counter()
    ep = dict(positions=points[selected], normals=grid.tool_normals[selected],
        ray_ids=selected, distances=grid.surface_distance[selected],
        owners=np.zeros(len(selected), np.int64),
        tool_ids=np.full(len(selected), sweep.tool_id, np.int32))
    descriptor = SimpleNamespace(**{name: getattr(grid, name) for name in (*FIELDS, 'tool_normals')})
    new = _snapshot(ep, descriptor, [command], surface_groups=np.array([group]))
    new.material_surface_ids = new.gaussian.attributes['surface_id'][:, 0]+3*(material_group-group)
    new.endpoints['owners'] = np.full(len(selected), block, np.int64)
    new.endpoints['ray_ids'] = selected+ray_offset+(block+1)*RAY_STRIDE
    for array in (*new.endpoints.values(), new.material_surface_ids):
        array.flags.writeable = False
    stats = dict(adaptive_seed_ms=(ready-started)*1000,
        adaptive_cut_ms=(evaluated-ready)*1000,
        adaptive_endpoint_readback_bytes=0,
        adaptive_seed_control_readback_bytes=evidence.nbytes + np.dtype(np.int64).itemsize,
        adaptive_gs_hit_queries=0,
        adaptive_analytic_candidates=len(candidates),
        adaptive_analytic_air=int(np.count_nonzero(evidence < 0)),
        adaptive_analytic_unknown=int(np.count_nonzero(evidence == 0)),
        adaptive_total_ms=(perf_counter()-started)*1000)
    return new, descriptor, stats

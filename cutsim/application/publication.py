"""Completed immutable endpoints and display ownership; no GUI imports."""
from dataclasses import dataclass
from copy import copy, deepcopy


@dataclass(frozen=True)
class ComputeEvent:
    kind: str
    completed: int
    message: str = ''
    result: object = None
    metrics: object = None


def display_material(material):
    """A private query/diagnostic context over immutable material history."""
    result = material.snapshot()
    # BoundaryWork contains mutable caches, unlike the persistent spatial root.
    if hasattr(result, '_boundary_work'):
        del result._boundary_work
    if hasattr(result, '_gs_surface_provenance'):
        from cutsim.core.surface_provenance import SurfaceProvenance
        if not isinstance(result._gs_surface_provenance,SurfaceProvenance):
            result._gs_surface_provenance = dict(result._gs_surface_provenance)
    return result


@dataclass(frozen=True)
class CompletedBlock:
    delta: object
    material: object
    attributes: object
    grid_cell: float

    @classmethod
    def capture(cls, sim, delta, attributes):
        # Geometry endpoints are persistent. The playback cursor and diagnostics
        # are not: give the consumer its own delta wrapper and material context.
        visible = copy(delta)
        from cutsim.core.replay_archive import replay_archive_enabled
        if not replay_archive_enabled():
            visible.__dict__.pop('_playback', None)
            visible.stats = deepcopy(delta.stats)
            return cls(visible, display_material(sim.material), attributes, sim.grid_cell)
        # Rendering/debug playback needs the completed clouds, not the cutting
        # work graph, spatial stores or the chain of old computation states.
        for name in ('_playback', '_local_final_state', '_precut_material',
                     'strip_intersections', '_archive_reference', '_archive_endpoints'):
            visible.__dict__.pop(name, None)
        from cutsim.core.splat_cloud import SplatCloud
        clouds = {}
        for name, value in tuple(visible.__dict__.items()):
            if isinstance(value, SplatCloud):
                if id(value) not in clouds:
                    detached = copy(value)
                    detached.__dict__.pop('_local_state', None)
                    clouds[id(value)] = detached
                setattr(visible, name, clouds[id(value)])
        visible.stats = deepcopy(delta.stats)
        return cls(visible, display_material(sim.material), attributes, sim.grid_cell)


class DisplayState:
    """GUI-owned published history; deliberately has no cutting methods/indexes."""
    def __init__(self, sim, disk_history=False):
        self.program = tuple(sim.program)
        self.stock_lo, self.stock_hi = sim.stock_lo.copy(), sim.stock_hi.copy()
        self.grid_sizing = copy(sim.grid_sizing)
        self.grid_cell = sim.grid_cell
        self.gs_density = getattr(sim, 'gs_density', 1.)
        self.shape_update_mode = getattr(sim, 'shape_update_mode', 'strip')
        self.use_splat_surface_grid = sim.use_splat_surface_grid
        self.splats = sim.splats
        self.material = display_material(sim.material)
        from cutsim.core.history_store import HistorySequence
        from cutsim.core.replay_archive import ReplaySequence, replay_archive_enabled
        self.deltas = (ReplaySequence() if replay_archive_enabled() else
                       HistorySequence() if disk_history else [])
        self.last_stats = None
        self.block_metrics = {}
        for delta in sim.deltas:
            visible = copy(delta)
            visible.__dict__.pop('_playback', None)
            visible.stats = deepcopy(delta.stats)
            self.deltas.append(visible)
        if self.deltas:
            self.last_stats = self.deltas[-1].stats

    def publish(self, block):
        if block.delta.block_index != len(self.deltas):
            raise ValueError('Completed blocks must be published in program order')
        self.deltas.append(block.delta)
        self.material = block.material
        self.splats = block.delta.playback().after
        self.grid_cell = block.grid_cell
        self.last_stats = block.delta.stats

    def record_metrics(self, index, metrics):
        """Keep small timing records, not worker graphs or archive objects."""
        from math import isfinite
        if not 0 <= index < len(self.deltas):
            raise ValueError('Timings require a published block')
        self.block_metrics[index] = {key:float(value) for key,value in metrics.items()
            if isinstance(key,str) and key.endswith('_seconds') and
            isinstance(value,(int,float)) and isfinite(value) and value>=0.}

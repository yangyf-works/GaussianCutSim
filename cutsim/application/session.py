"""Single-owner block pipeline used by the desktop worker and release checks.

Cutting, presentation preparation, durable replay and publication have distinct
boundaries. A block is publishable only after all stages succeed. An exception
poisons this session: never retry a partially mutated engine silently.
"""
from time import perf_counter

from cutsim.application.publication import CompletedBlock
from cutsim.core.history_store import history_enabled, checkpoint_geometry, HistorySequence
from cutsim.core.replay_archive import ReplayArchive, ReplaySequence
from cutsim.diagnostics.memory import process_memory
from cutsim.rendering.playback import PlaybackAttributeCache
from cutsim.rendering.render_cache import RenderAttributeCache


class ComputeSession:
    def __init__(self, engine, archive_path=None):
        self.engine = engine
        self.archive = ReplayArchive(archive_path) if archive_path is not None else None
        self.disk_history = history_enabled()
        self.attributes = PlaybackAttributeCache(disk_history=self.disk_history and self.archive is None)
        self.cache = RenderAttributeCache()
        if self.archive is not None:
            engine.deltas = ReplaySequence(engine.deltas)
        elif self.disk_history:
            engine.deltas = HistorySequence(engine.deltas)
        self.completed = len(engine.deltas)
        self.failed = False
        self.stage = 'idle'

    def check_frontier(self, expected):
        if self.failed:
            raise RuntimeError('Reset calculation after a worker failure')
        if expected != self.completed or len(self.engine.deltas) != self.completed:
            raise RuntimeError('Compute worker and displayed history disagree; reset calculation')

    def next(self, settings, preparing=None):
        self.check_frontier(self.completed)
        started = perf_counter()
        try:
            self.stage = 'checkpoint'
            stage_start = perf_counter()
            if self.archive is not None or self.disk_history:
                checkpoint_geometry(self.engine, render_cache=self.cache)
            checkpoint_seconds = perf_counter() - stage_start

            self.stage = 'cutting'
            stage_start = perf_counter()
            configure_coverage = getattr(self.engine, 'set_coverage_display_scale', None)
            if configure_coverage is not None:
                configure_coverage(settings.sigma_scale)
            delta = self.engine.compute_next()
            cutting_seconds = perf_counter() - stage_start
            if delta is None:
                if len(self.engine.deltas) != self.completed:
                    raise RuntimeError('Engine changed history without returning a block')
                self.stage = 'idle'
                return None
            if delta.block_index != self.completed or len(self.engine.deltas) != self.completed + 1:
                raise RuntimeError('Engine must produce exactly one ordered block')

            self.stage = 'preparing'
            if preparing is not None:
                preparing(self.completed + 1)
            stage_start = perf_counter()
            self.attributes.prepare_computed_block(delta, self.engine.material, settings, self.cache)
            render_seconds = perf_counter() - stage_start
            stage_start = perf_counter()
            block = CompletedBlock.capture(self.engine, delta, self.attributes.export())
            capture_seconds = perf_counter() - stage_start

            self.stage = 'archiving'
            stage_start = perf_counter()
            if self.archive is not None:
                reference = self.archive.write(block)
                self.engine.deltas.archived(reference)
                self.attributes.release_published()
                block = reference
            archive_seconds = perf_counter() - stage_start
            metrics = dict(cutting_seconds=cutting_seconds,
                           preparation_seconds=render_seconds + capture_seconds,
                           render_preparation_seconds=render_seconds,
                           publication_capture_seconds=capture_seconds,
                           completed_before_reused=self.attributes.completed_before_reused,
                           checkpoint_seconds=checkpoint_seconds, archive_seconds=archive_seconds,
                           pipeline_seconds=perf_counter() - started,
                           worker_memory=process_memory())
            if self.archive is not None:
                metrics['archive'] = self.archive.metrics()
            preparation_stats=getattr(self.attributes,'preparation_stats',None)
            if isinstance(preparation_stats,dict):
                metrics['render_update']=preparation_stats.copy()
            self.completed += 1
            self.stage = 'idle'
            return block, metrics
        except Exception:
            self.failed = True
            raise

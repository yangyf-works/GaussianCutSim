"""GaussianCutSim diagnostics / profiling implementation."""
from __future__ import annotations

from cutsim._version import __version__

import numpy as np
import time


def install_performance_logging(detailed=False):
    import cProfile
    import functools
    import itertools
    import json
    import logging
    from logging.handlers import RotatingFileHandler
    from pathlib import Path
    import pstats
    import time

    if globals().get("_CUTSIM_PERF_INSTALLED", False):
        return

    log_path = Path(__file__).resolve().parents[2] / "CutSim_performance.jsonl"
    logger = logging.getLogger("cutsim.performance")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024,
        backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    sequence = itertools.count(1)

    def emit(data):
        data["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(json.dumps(data, ensure_ascii=False))

    def cache_counts(obj):
        return {
            name: int(getattr(obj, name, 0))
            for name in ("hits", "misses", "recolors")
        }

    def wrap(func, label, detailed=False):
        @functools.wraps(func)
        def measured(*args, **kwargs):
            call_id = next(sequence)
            profiler = cProfile.Profile() if detailed else None
            before = (
                cache_counts(args[0])
                if label == "render_cache" else None)
            result = None
            error = None

            emit({"event": "start", "id": call_id, "operation": label})
            if label in ("render_pack", "compute_next"):
                print(f"[PERF #{call_id}] {label} START", flush=True)

            def spatial_counts():
                mi=1 if label=='render_pack' else 2
                material=(getattr(args[0],'material',None) if label=='compute_next' and args
                          else (args[mi] if len(args)>mi else kwargs.get('material')))
                index=getattr(material,'_sweep_index',None)
                return dict(sdf_requested=getattr(material,'last_sweep_sdf_requested',0),
                            sdf_evaluated=getattr(material,'last_sweep_sdf_evaluated',0),
                            sdf_tree_nodes=getattr(material,'last_sweep_index_nodes',0),
                            sdf_bound_pairs=getattr(material,'last_sweep_index_pairs',0),
                            aabb_tree_nodes=getattr(index,'query_nodes',0),
                            aabb_queries=getattr(index,'query_calls',0))
            spatial_before=spatial_counts()
            start = time.perf_counter()
            try:
                if profiler is not None:
                    profiler.enable()
                result = func(*args, **kwargs)
                return result
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                if profiler is not None:
                    profiler.disable()

                elapsed_ms = (time.perf_counter() - start) * 1000.0
                data = {
                    "event": "end",
                    "id": call_id,
                    "operation": label,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "error": error,
                }

                # Instrumentation errors must not interrupt the simulation.
                try:
                    if label == "render_cache":
                        after = cache_counts(args[0])
                        data["cache_delta"] = {
                            key: after[key] - before[key]
                            for key in before
                        }

                    if label in ("render_pack", "render_cache"):
                        index = 0 if label == "render_pack" else 1
                        cloud = (
                            args[index] if len(args) > index
                            else kwargs.get("cloud"))
                        mi = 1 if label == 'render_pack' else 2
                        mat = args[mi] if len(args) > mi else kwargs.get('material')
                        edge_stats = getattr(mat, 'last_edge_split_stats', {})
                        data['edge_phase_ms'] = getattr(mat,'last_edge_phase_ms',{}).copy()
                        data['edge_update'] = {k: v for k, v in edge_stats.items()
                                               if k.startswith('edge_')}
                        data['local_update']={k:v for k,v in edge_stats.items() if k.startswith('local_')}
                        data['native_clip']={k:v for k,v in edge_stats.items() if k.startswith('clip_')}
                        if cloud is not None:
                            data["input_points"] = len(cloud.positions)

                        if isinstance(result, dict):
                            data["rendered_points"] = len(
                                result.get("positions", ()))
                            if 'clip_pending' in result:
                                data['prepared_points']=len(result['positions'])
                                data['pending_points']=int(result['clip_pending'].sum())
                                data['rendered_parents']=data['prepared_points']-data['pending_points']
                                joins=result.get('clip_concave')
                                data['concave_draw_duplicates']=int(joins[~result['clip_pending']].sum()) if joins is not None else 0
                                data['rendered_points']=data['rendered_parents']+data['concave_draw_duplicates']
                            source = result.get("source_ids")
                            if source is not None and len(source):
                                import numpy as np
                                _, counts = np.unique(
                                    source, return_counts=True)
                                data["max_children_per_parent"] = int(
                                    counts.max())
                                data["parents_over_3_children"] = int(
                                    np.count_nonzero(counts > 3))

                    if label == "compute_next" and result is not None:
                        stats = getattr(result, "stats", {})
                        data['point_spatial_update']={k:stats[k] for k in (
                            'retire_index_cells','retire_index_bucket_points','retire_sdf_points',
                            'retire_total_points','changed_spatial_cells','split_stock_parents','split_machined_parents') if k in stats}
                        data["cut_stage_ms"] = {
                            key: float(value)
                            for key, value in stats.items()
                            if key.endswith("_ms")
                        }

                    if profiler is not None:
                        records = pstats.Stats(profiler).stats
                        phases = []
                        for (filename, line, name), values in records.items():
                            primitive, calls, own_time, cumulative, _ = values
                            phases.append({
                                "function": name,
                                "file": Path(filename).name,
                                "line": line,
                                "calls": calls,
                                "self_ms": round(own_time * 1000, 3),
                                "inclusive_ms": round(cumulative * 1000, 3),
                            })
                        phases.sort(
                            key=lambda item: item["inclusive_ms"],
                            reverse=True)
                        data["top_functions"] = phases[:30]

                    spatial_after=spatial_counts()
                    data['spatial_query_delta']={k:spatial_after[k]-v for k,v in spatial_before.items()}
                    emit(data)
                    suffix = (
                        f" cache={data['cache_delta']}"
                        if "cache_delta" in data else "")
                    update = data.get('edge_update', {})
                    if 'edge_recomputed' in update:
                        suffix += (f" edge_reused={update.get('edge_reused',0)}"
                                   f" edge_recomputed={update['edge_recomputed']}")
                    if label == 'render_pack' and data.get('edge_phase_ms'):
                        phases=data['edge_phase_ms']
                        suffix += " phases_ms={"+", ".join(
                            f"{k[:-3]}:{v:.1f}" for k,v in phases.items() if k!='total_ms')+"}"
                        suffix += (f" shape_excluded={update.get('edge_material_shape_excluded',0)}"
                                   f" shape_reused={update.get('edge_reused_shape_filter',0)}")
                        if 'edge_candidate_recomputed' in update:
                            suffix += (f" candidate_reused={update['edge_candidate_reused']}"
                                       f" candidate_recomputed={update['edge_candidate_recomputed']}")
                    spatial=data['spatial_query_delta']
                    if label in ('render_pack','compute_next'):
                        suffix += (f" sdf_evaluated={spatial['sdf_evaluated']}/{spatial['sdf_requested']}"
                                   f" aabb_nodes={spatial['aabb_tree_nodes']}")
                    local=data.get('local_update',{})
                    if local:
                        phase_values=data.get('edge_phase_ms',{})
                        if phase_values:
                            suffix += " phases_ms={"+", ".join(f"{k[:-3]}:{v:.1f}" for k,v in phase_values.items() if k!='total_ms')+"}"
                        suffix += (f" local_updated={local.get('local_updated_points',0)}/{local.get('local_total_points',0)}"
                                   f" reference_only={local.get('local_reference_points',0)}"
                                   f" prepared={local.get('local_prepared_points',0)}"
                                   f" full={local.get('local_full_rebuild',0)}")
                    print(
                        f"[PERF #{call_id}] {label}: "
                        f"{elapsed_ms:.1f} ms{suffix}",
                        flush=True)
                except Exception as log_error:
                    print(
                        f"[PERF] Logging failed: {log_error}",
                        flush=True)

        return measured

    from cutsim.rendering import gaussian_attributes
    from cutsim.rendering.render_cache import RenderAttributeCache
    from cutsim.cutting.realtime_simulator import SweepDrivenRealtimeSimulator
    from cutsim.ui.debug_gui import SweepDrivenRealtimeGUI

    gaussian_attributes.pack_gaussian_attributes = wrap(
        gaussian_attributes.pack_gaussian_attributes, "render_pack", detailed=detailed)
    targets = [
        (RenderAttributeCache, "get", "render_cache"),
        (SweepDrivenRealtimeSimulator, "compute_next", "compute_next"),
        (SweepDrivenRealtimeGUI, "_redraw", "redraw"),
    ]
    for cls, method_name, label in targets:
        method = getattr(cls, method_name, None)
        if callable(method):
            setattr(cls, method_name, wrap(method, label))

    globals()["_CUTSIM_PERF_INSTALLED"] = True
    emit({"event": "logging_enabled", "path": str(log_path), "version": __version__, "detailed": detailed})
    print(f"[PERF] Log: {log_path}", flush=True)

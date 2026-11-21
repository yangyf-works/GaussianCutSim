"""GaussianCutSim diagnostics / stats implementation."""
from __future__ import annotations

import numpy as np


def resolution_histogram(support: np.ndarray):
    h = np.asarray(support, dtype=np.float64).reshape(-1)
    if not len(h):
        return tuple()
    rounded = np.round(h, 8)
    values, counts = np.unique(rounded, return_counts=True)
    order = np.argsort(values)[::-1]
    return tuple((float(values[i]), int(counts[i])) for i in order)






def print_rt_stats(s):
    print(f"[RT block {s['step']:02d}] {s['tool_type']} D={s['tool_diameter']:.2f} move={s['move_length']:.2f} h={s['base_h']:.4f} | "
          f"cand={s['candidate_count']:,} direct={s['direct_count']:,} fallback={s['fallback_candidates']:,} discard={s['discard_count']:,} | "
          f"grids={s['fallback_patches']} nodes={s['fallback_grid_nodes']:,} ring={s.get('footprint_ring_candidates',0):,} | "
          f"unresolvedToGrid={s.get('direct_unresolved_to_grid',0):,} | "
          f"splitStock={s.get('split_stock_parents',0):,} splitMachined={s.get('split_machined_parents',0):,} | "
          f"repr simple={s.get('simple_clip_candidates',0):,} complex={s.get('complex_candidates',0):,} tri={s.get('vertex_grid_triggers',0):,} | rejectExisting={s.get('direct_existing_rejected',0):,} covered={s.get('direct_covered_dropped',0):,} | old-{s['removed_old']:,} new+{s['added_total']:,} | "
          f"expand={s['candidate_expand_ms']:.2f} classify={s['classify_ms']:.2f} grid={s['fallback_grid_ms']:.2f} triDetect={s.get('vertex_grid_ms',0.0):.2f} retire={s['retire_ms']:.2f} total={s['total_ms']:.2f} ms")

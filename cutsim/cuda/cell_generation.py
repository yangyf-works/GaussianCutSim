"""Evaluate every supported profile cell at 2x2 samples, then coarsen safely.

Three or four positive children permit a parent GS only if its own centre is
material. Otherwise retain the positive children, including thin 1/2-hit cuts.
Classification and voting stay on CUDA; only emitted child rays are published.
"""
from time import perf_counter
from types import SimpleNamespace
import numpy as np

from cutsim.cuda.adaptive_profile import CHARTS, FIELDS, _generate
from cutsim.cuda.addition_refinement import _selected_template


def short_move_sampling(command, atlas):
    """Use one base sampling pitch as the short-move threshold, in 3D mm.

    Atlas pitches already include tool sampling settings and GS density.
    A stationary command has no newly traversed band and is not a short move.
    """
    from cutsim.core.commands import command_endpoints
    a, b = command_endpoints(command)
    length = float(np.linalg.norm(b-a))
    pitches = np.asarray(atlas.pitches)
    positive = pitches[np.isfinite(pitches) & (pitches > 0)]
    pitch = float(positive.min()) if len(positive) else 0.
    return 0. < length <= pitch, length, pitch


def cell_choices(votes, parent_material, source, child_material):
    """Array-module independent admission rule; no parent/child duplication."""
    coarse = (votes >= 3) & parent_material
    children = child_material & ~coarse[source]
    return coarse, children


def generate_cells(model, command, sweep, block, group, material_group,
                   stock_bounds, *, batch_rays=1024):
    """A batch limit bounds temporary rays, never the cells that get evaluated.

    Non-parametric ramp envelope samples have no 2D cell subdivision mapping;
    preserve their original centre predicate and report them explicitly.
    """
    from cutsim.cuda.adaptive_device import prepare_device, generate_tool_gs_device
    from cutsim.cuda.analytic_ray_generation import select_generation
    from cutsim.cuda.material_evidence import finite_material_evidence_device

    started = perf_counter()
    surface = model.surface; rt = surface.rt; cp = rt.cp
    atlas = model.grid; template = getattr(atlas, 'profile_template', atlas)
    count = len(template.origins)
    supported = np.isin(template.charts, CHARTS)
    parents = np.flatnonzero(supported)
    capacity = max(1, batch_rays // 4)
    box = atlas.local_boxes[block]
    margin = getattr(model, 'ray_margin', None)
    model.sweep = sweep
    pieces = []; evaluated = 0; batches = 0
    with rt.context():
        votes = cp.zeros(count, cp.int32)
        totals = cp.zeros(3, cp.int64)  # material, air, unknown (valid rays only)
        for start in range(0, len(parents), capacity):
            ids = parents[start:start+capacity]
            compact = _selected_template(template, ids)
            raw_model = SimpleNamespace(_origins=cp.asarray(compact.origins),
                                        _directions=cp.asarray(compact.directions))
            raw, source, _ = _generate(surface, raw_model, compact,
                np.arange(len(ids), dtype=np.int64), None, None, command, block,
                4*len(ids), risk_mask=np.ones(len(ids), bool), divisions=2)
            batches += 1
            if raw is None:
                continue
            source = cp.asarray(ids)[source]
            packet, valid = prepare_device(surface, raw, sweep, box, stock_bounds, margin)
            evidence, _, _ = finite_material_evidence_device(surface,
                packet[valid, 3:6], packet[valid, :3])
            packet[:, 6] = -2
            packet[valid, 6] = evidence
            totals += cp.stack([(evidence == value).sum() for value in (1, -1, 0)])
            positive = cp.flatnonzero(packet[:, 6] == 1)
            cp.add.at(votes, source[positive], 1)
            evaluated += len(packet)
            if len(positive):
                pieces.append((SimpleNamespace(**{k:cp.ascontiguousarray(getattr(raw,k)[positive])
                    for k in FIELDS}), source[positive], cp.ascontiguousarray(packet[positive])))

        # Only potential coarse GS need a centre query. Ramp samples retain
        # their existing rule; an unevaluated parent is never called material.
        select_generation(model, (votes >= 3) | cp.asarray(~supported))
        parent_material = model.added != 0
        model.added[:] = ((votes >= 3) | cp.asarray(~supported)) & parent_material
        model.selected = cp.flatnonzero(model.added)
        descriptor = extra = None
        published = readback = 0
        if pieces:
            raw = SimpleNamespace(**{k:cp.concatenate([getattr(p[0],k) for p in pieces]) for k in FIELDS})
            source = cp.concatenate([p[1] for p in pieces])
            packet = cp.concatenate([p[2] for p in pieces])
            _, keep = cell_choices(votes, parent_material, source, packet[:, 6] == 1)
            selected = cp.flatnonzero(keep)
            if len(selected):
                # Publication IDs must not depend on the temporary batch size.
                selected = selected[cp.lexsort(cp.stack((selected, source[selected])))]
                raw = SimpleNamespace(**{k:cp.ascontiguousarray(getattr(raw,k)[selected]) for k in FIELDS})
                extra, grid, host_parents, publication = generate_tool_gs_device(surface,
                    raw, cp.ascontiguousarray(source[selected]), command, sweep, block, group,
                    material_group, box, count, margin, stock_bounds=stock_bounds,
                    evaluated_packet=cp.ascontiguousarray(packet[selected]))
                for value in (*vars(grid).values(), host_parents):
                    if isinstance(value, np.ndarray):value.flags.writeable = False
                descriptor = dict(grid=grid, parents=host_parents, base_rays=count,
                                  replaced_parents=np.empty(0, np.int64))
                published = len(grid.origins)
                readback = publication['adaptive_publication_readback_bytes']
        # One small diagnostic readback, independent of the number of batches.
        counters = cp.asnumpy(cp.concatenate((totals,
            cp.stack([(votes[cp.asarray(supported)] == i).sum() for i in range(5)]),
            cp.stack([((votes >= 3) & ~parent_material).sum()]))), stream=rt.stream)
        rt.stream.synchronize()
    elapsed = (perf_counter()-started)*1000
    model.stats.update(generation_policy='all_cells_2x2', last_device_step_s=elapsed/1000)
    stats = dict(addition_refinement_enabled=True, addition_refinement_divisions=2,
        addition_refinement_selection='all_cells_2x2', addition_refinement_passes=int(bool(len(parents))),
        addition_refinement_selected_cells=len(parents), addition_refinement_unsupported_cells=int((~supported).sum()),
        addition_refinement_budget_skipped_cells=0, addition_refinement_batch_rays=max(4,batch_rays),
        addition_refinement_batches=batches, addition_refinement_evaluated_rays=evaluated,
        addition_refinement_rays=published, addition_refinement_added=published,
        addition_refinement_material_points=int(counters[0]), addition_refinement_air_points=int(counters[1]),
        addition_refinement_unknown_points=int(counters[2]),
        addition_refinement_vote_cells=[int(v) for v in counters[3:8]],
        addition_refinement_parent_rejected_cells=int(counters[8]),
        addition_refinement_intermediate_readback_bytes=0,
        addition_refinement_diagnostic_readback_bytes=counters.nbytes,
        addition_refinement_publication_readback_bytes=readback, addition_refinement_ms=elapsed)
    return extra, descriptor, stats

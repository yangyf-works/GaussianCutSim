"""Prime CUDA kernels in the compute worker using disposable geometry."""
from contextlib import ExitStack
from time import perf_counter


def warmup_program():
    from cutsim.core.commands import CutCommand, CuttingRange
    program = []
    paths = (
        ('flat', ((-1., -.6, .5), (-1., -.6, -.3), (0., -.6, -.3),
                  (1., -.6, -.6), (1., -.6, .5))),
        ('ball', ((0., -1., .5), (0., -1., -.3), (0., 0., -.6),
                  (0., 1., -.6), (0., 1., .5))),
    )
    for tool_id, (kind, points) in enumerate(paths, 1):
        for a, b in zip(points, points[1:]):
            program.append(CutCommand(a, b, .5, tool_id=tool_id, tool_type=kind,
                                      cutting_range=CuttingRange(0., 1.5)))
    return program


def warmup_cuda(settings, cancel, progress):
    """Return completion/timing; never receive or modify the user's simulator.

    Cancellation is checked between small moves. A cancelled pass is retried
    on the next request, with already compiled kernels still cached.
    """
    from cutsim.core.cuda_cache import configure_cuda_cache
    configure_cuda_cache()
    if cancel.is_set():
        return dict(complete=False, completed=0, total=8, seconds=0.)
    from cutsim.cutting.realtime_simulator import SweepDrivenRealtimeSimulator
    from cutsim.rendering.playback import PlaybackAttributeCache
    from cutsim.rendering.render_cache import RenderAttributeCache
    program = warmup_program()
    started = perf_counter()
    completed = 0
    with ExitStack() as cleanup:
        sim = SweepDrivenRealtimeSimulator(program=program, initial_splat=.5,
            stock_lo=(-2., -2., -1.5), stock_hi=(2., 2., 0.))
        cleanup.callback(sim.close)
        attributes = PlaybackAttributeCache(disk_history=False)
        cleanup.callback(attributes.clear)
        cache = RenderAttributeCache()
        cleanup.callback(cache.clear)
        for index, command in enumerate(program):
            if cancel.is_set():
                break
            progress(index, len(program), command.tool_type)
            delta = sim.compute_next()
            attributes.prepare_computed_block(delta, sim.material, settings, cache)
            completed += 1
    return dict(complete=completed == len(program), completed=completed,
                total=len(program), seconds=perf_counter()-started)

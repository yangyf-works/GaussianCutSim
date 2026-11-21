"""Block-boundary compute scheduling independent of Qt and Open3D."""
from queue import Empty, Queue
from threading import Event, Thread
from cutsim.application.publication import ComputeEvent
from operator import index


def validate_count(count):
    if isinstance(count,bool):raise ValueError('Block count must be a nonnegative integer')
    try:value=index(count)
    except TypeError as exc:raise ValueError('Block count must be a nonnegative integer') from exc
    if value<0:raise ValueError('Block count must be a nonnegative integer')
    return value


class ComputeJob:
    """Single writer for the engine, separate from the GUI's DisplayState.

    Completed result ownership crosses the queue at block boundaries. Cancellation completes the
    current block; interrupting compute_next would leave partial cutting state.
    A failed job requires an engine reset; already published blocks remain usable.
    """

    def __init__(self):
        self._events = Queue()
        self._cancel = Event()
        self._thread = None

    def start(self, sim, count, prepare_block=None):
        count=validate_count(count)
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError('A computation is already running')
        self._cancel.clear()
        self.drain()
        self._thread = Thread(target=self._run, args=(sim, count, prepare_block),
                              name='cutsim-compute', daemon=False)
        self._thread.start()

    def _run(self, sim, count, prepare_block=None):
        completed = len(sim.deltas)
        try:
            for _ in range(count):
                if self._cancel.is_set():
                    break
                self._events.put(ComputeEvent('working', completed))
                delta = sim.compute_next()
                if delta is None:
                    break
                computed = len(sim.deltas)
                result = None
                if prepare_block is not None:
                    self._events.put(ComputeEvent('preparing', computed))
                    result = prepare_block(delta)
                completed = computed
                self._events.put(ComputeEvent('progress', completed, result=result))
            self._events.put(ComputeEvent('done', completed,
                                          'Stopped after block' if self._cancel.is_set() else 'Calculation complete'))
        except Exception as exc:
            self._events.put(ComputeEvent('error', completed, f'{type(exc).__name__}: {exc}'))

    def cancel(self):
        self._cancel.set()

    def drain(self):
        events = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except Empty:
                # A terminal event is posted as the worker's last operation.
                # Join here so an immediate new job cannot race thread teardown.
                if any(event.kind in ('done', 'error') for event in events):
                    self._thread.join()
                return events

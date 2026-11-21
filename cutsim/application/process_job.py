"""Persistent compute worker with its own interpreter; no Qt/Open3D in the worker.

Only completed block snapshots cross the pipe. A receiver thread does transport,
unpickling and playback indexing before publishing to the GUI's local queue.
The worker retains CUDA kernels across preparation, calculation and resets.
"""
import multiprocessing as mp
import os
from threading import Thread
from queue import Queue
from time import perf_counter
from cutsim.core.replay_archive import ArchiveOwner, ReplayReference, replay_archive_enabled

from cutsim.application.jobs import ComputeJob,validate_count
from cutsim.application.publication import ComputeEvent


def _worker(connection,cancel):
    from cutsim.application.session import ComputeSession
    from cutsim.application.warmup import warmup_cuda
    session=None
    warmed=False
    try:
        while True:
            request=connection.recv()
            if request[0]=='close':return
            try:
                if request[0]=='prime':
                    preparation=warmup_cuda(request[1],cancel,lambda *args: None)
                    warmed=preparation['complete']
                    connection.send(ComputeEvent('done',0,metrics=dict(
                        warmup_complete=warmed,worker_pid=os.getpid())))
                    continue
                if request[0]=='initialize':
                    if session is not None:
                        session.engine.close()
                    session=ComputeSession(request[1],request[2] if len(request)>2 else None)
                    continue
                action,count,settings,expected=request
                if action not in ('warmup','compute'):
                    raise ValueError(f'Unknown compute request: {action}')
                if session is None:raise RuntimeError('Compute worker is not initialized')
                session.check_frontier(expected)
                warmup_seconds=0.
                if not warmed:
                    session.stage='warmup'
                    def progress(index,total,kind):
                        connection.send(ComputeEvent('warming',session.completed,
                            f'Preparing calculation: {index+1}/{total} ({kind.title()})...',
                            metrics=dict(warmup_completed=index,warmup_total=total)))
                    preparation=warmup_cuda(settings,cancel,progress)
                    warmed=preparation['complete']
                    warmup_seconds=preparation['seconds']
                    session.stage='idle'
                    if not warmed:
                        connection.send(ComputeEvent('done',session.completed,
                            'Preparation stopped. Compute all to continue preparation and calculation.',
                            metrics=dict(warmup_complete=False,warmup_seconds=warmup_seconds,worker_pid=os.getpid())))
                        continue
                if action=='warmup':
                    connection.send(ComputeEvent('done',session.completed,'Calculation ready.',
                        metrics=dict(warmup_complete=True,warmup_seconds=warmup_seconds,worker_pid=os.getpid())))
                    continue
                for _ in range(count):
                    if cancel.is_set():break
                    connection.send(ComputeEvent('working',session.completed))
                    result=session.next(settings,preparing=lambda n:connection.send(ComputeEvent('preparing',n)))
                    if result is None:break
                    block,metrics=result
                    connection.send(ComputeEvent('progress',session.completed,result=block,metrics=metrics))
                    del block,result
                connection.send(ComputeEvent('done',session.completed,
                    'Stopped after block' if cancel.is_set() else 'Calculation complete',
                    metrics=dict(warmup_complete=True,warmup_seconds=warmup_seconds,worker_pid=os.getpid())))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                if session is not None:session.failed=True
                connection.send(ComputeEvent('error',session.completed if session is not None else 0,
                    f'{type(exc).__name__}: {exc}',metrics={'failed_stage':session.stage if session is not None else 'initialize'}))
    except (EOFError,BrokenPipeError):
        pass
    finally:
        try:
            if session is not None:session.engine.close()
        finally:connection.close()


class ProcessComputeJob(ComputeJob):
    """Keep GUI-side engine configuration separate from remote evolving state."""
    def __init__(self):
        super().__init__()
        # A fast CPU worker must not accumulate decoded blocks while the user
        # is interacting with a slower viewport. Remaining results stay on disk.
        self._events=Queue(maxsize=4)
        self._context=mp.get_context('spawn')
        self._cancel=self._context.Event()
        self._process=self._connection=None
        self._source=None
        self.completed=0
        self._failed=False
        self._archive_owner=None
        self._startup_thread=None
        self._startup_result=None
        self._startup_reported=False

    @property
    def startup_running(self):
        return self._startup_thread is not None and self._startup_thread.is_alive()

    def start_background_warmup(self):
        """Prime before loading user data/UI, without owning a simulation."""
        if self._startup_thread is not None or self._process is not None:
            return
        self._startup_thread=Thread(target=self._prime, name='cutsim-startup', daemon=False)
        self._startup_thread.start()

    def _prime(self):
        try:
            from cutsim.rendering.settings import RenderSettings
            self._open_process()
            self._connection.send(('prime',RenderSettings()))
            self._startup_result=self._connection.recv()
        except Exception as exc:
            self._startup_result=ComputeEvent('error',0,f'Compute worker: {type(exc).__name__}: {exc}')

    def drain(self):
        events=super().drain()
        if (not self.startup_running and self._startup_result is not None
                and not self._startup_reported
                and (self._thread is None or not self._thread.is_alive())):
            self._startup_reported=True
            if self._startup_result.kind=='error':
                self._failed=True
                events.append(self._startup_result)
        return events

    def _open_process(self):
        parent,child=self._context.Pipe()
        self._process=self._context.Process(target=_worker,args=(child,self._cancel),
                                            name='cutsim-compute',daemon=True)
        try:self._process.start()
        except BaseException:
            parent.close();child.close();self._process=None
            raise
        child.close();self._connection=parent

    def start_prepared(self,sim,count,settings,expected):
        self._start_request(sim,count,settings,expected,'compute')

    def start_warmup(self,sim,settings,expected=0):
        """Prepare in the same worker that will later compute real blocks."""
        self._start_request(sim,0,settings,expected,'warmup')

    def _start_request(self,sim,count,settings,expected,action):
        count=validate_count(count)
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError('A computation is already running')
        source=(sim,sim.material)
        if self._failed and self._source==source:
            raise RuntimeError('Reset calculation after a worker failure')
        # Leave an unreported startup error for _receive to publish with this
        # request; do not consume it silently while accepting Compute all.
        self._cancel.clear();super().drain()
        # Starting the process and serializing even a large initial state must
        # not run in a Qt/Open3D event callback.
        self._thread=Thread(target=self._receive,args=(source,count,settings,expected,action),
                            name='cutsim-results',daemon=False)
        self._thread.start()

    def _receive(self,source,count,settings,expected,action='compute'):
        terminal=None
        try:
            if self._startup_thread is not None:
                self._startup_thread.join()
                if not self._startup_reported and self._startup_result.kind=='error':
                    self._startup_reported=True
                    raise RuntimeError(self._startup_result.message)
                self._startup_reported=True
            if self._process is None or not self._process.is_alive() or self._failed:
                self._close_process()
                self._open_process()
            if self._source!=source:
                self._archive_owner=ArchiveOwner() if replay_archive_enabled() else None
                self._connection.send(('initialize',source[0],
                             self._archive_owner.path if self._archive_owner else None))
                self._source=source;self._failed=False
            self._connection.send((action,count,settings,expected))
            while True:
                event=self._connection.recv()
                if event.result is not None:
                    if isinstance(event.result,ReplayReference):
                        started=perf_counter()
                        reference=event.result
                        reference.owner=self._archive_owner
                        event=ComputeEvent(event.kind,event.completed,event.message,reference.load(),event.metrics)
                        event.metrics['read_seconds']=perf_counter()-started
                    # Pickle reconstructs writable NumPy buffers. Restore the
                    # endpoint contract and prepare the consumer's private cursor.
                    for endpoint in (event.result.attributes.before,event.result.attributes.after):
                        from cutsim.rendering.chunked_attributes import freeze_endpoint
                        freeze_endpoint(endpoint)
                    started=perf_counter()
                    event.result.delta.playback().prepare_motion()
                    event.metrics['receiver_preparation_seconds']=perf_counter()-started
                self.completed=event.completed
                if event.kind in ('done','error'):
                    self._failed=event.kind=='error';terminal=event;break
                self._events.put(event)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._failed=True
            terminal=ComputeEvent('error',self.completed,f'Compute worker: {type(exc).__name__}: {exc}')
        finally:
            # Last action, so ComputeJob.drain joins no heavy receiver work.
            if terminal is not None:self._events.put(terminal)

    def _close_process(self):
        if self._connection is not None:
            try:self._connection.send(('close',))
            except (OSError,EOFError):pass
            self._connection.close();self._connection=None
        if self._process is not None:
            self._process.join(5.)
            if self._process.is_alive():
                self._process.terminate();self._process.join()
            self._process.close();self._process=None
        self._source=None
        self._archive_owner=None

    def close(self):
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError('Wait for preparation or calculation before closing its compute worker')
        if self.startup_running:
            self.cancel()
            self._startup_thread.join()
        self._close_process();self.completed=0;self._failed=False

    def shutdown(self):
        """Release startup/computation after app exit or a launch failure."""
        self.cancel()
        while self._thread is not None and self._thread.is_alive():
            self.drain()
            self._thread.join(.02)
        self.close()

    def reset_session(self):
        """Drop task ownership while preserving a healthy worker's CUDA cache."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError('Wait for preparation or calculation before resetting')
        if self._failed:
            # A failed CUDA operation may have poisoned the process context.
            self._close_process()
            self._startup_reported=True
        self._source=None;self._archive_owner=None
        self.completed=0;self._failed=False

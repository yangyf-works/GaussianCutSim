"""Current process resident/committed memory, without optional dependencies."""
import ctypes
import os


def process_memory():
    if os.name != 'nt': return {}
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[
            (name,ctypes.c_size_t) for name in ('peak','rss','paged_peak','paged',
                'nonpaged_peak','nonpaged','commit','peak_commit','private')]
    record=Counters();record.cb=ctypes.sizeof(record)
    current=ctypes.windll.kernel32.GetCurrentProcess
    current.restype=ctypes.c_void_p
    query=ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
    if not query(current(),ctypes.byref(record),record.cb):raise ctypes.WinError()
    return dict(working_set_bytes=record.rss,peak_working_set_bytes=record.peak,
                private_bytes=record.private)

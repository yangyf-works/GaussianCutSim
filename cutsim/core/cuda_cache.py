"""Configure a persistent kernel cache before importing CuPy."""
import os
from pathlib import Path
import sys
from uuid import uuid4


def configure_cuda_cache():
    """Respect overrides; never put the default cache in a temporary directory."""
    configured = os.environ.get('CUPY_CACHE_DIR')
    if configured:
        path = Path(os.path.expandvars(configured)).expanduser().resolve()
    elif sys.platform == 'win32':
        base = Path(os.environ.get('LOCALAPPDATA') or Path.home() / 'AppData' / 'Local')
        path = base / 'GaussianCutSim' / 'cuda-cache'
    else:
        base = Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache')
        path = base / 'GaussianCutSim' / 'cuda-cache'
    path = path.resolve()
    probe = path / f'.write-check-{uuid4().hex}'
    try:
        path.mkdir(parents=True, exist_ok=True)
        # Fail promptly instead of letting CuPy retry temporary-file creation.
        with probe.open('xb'):
            pass
        probe.unlink()
    except OSError as error:
        raise RuntimeError(f'CUDA cache is not writable: {path}. Set CUPY_CACHE_DIR to a writable directory.') from error
    os.environ['CUPY_CACHE_DIR'] = str(path)
    return path

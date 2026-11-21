"""GS ray/area query extent separate from rendering and clipping settings.

Scoped simulator calls use their support_sigma setting; the production
computation adapter selects 1.5 sigma. A snapshot's support_sigma takes
precedence over the current context. Queries without either setting use
the context default of 2 sigma. The radius is this factor times the GS sigma.
"""
from contextvars import ContextVar
from contextlib import contextmanager
from functools import wraps

_factor = ContextVar('gs_support_sigma', default=2.)


def validate_factor(value):
    value = float(value)
    if value not in (1.5, 2.):
        raise ValueError('GS support must be 1.5 or 2 sigma')
    return value


def support_factor(snapshot=None):
    return float(getattr(snapshot, 'support_sigma', _factor.get()))


def support_radius(snapshot):
    return support_factor(snapshot) * snapshot.gaussian.attributes['scale'][:, 0].astype(float)


@contextmanager
def support_scope(value):
    token = _factor.set(validate_factor(value))
    try:
        yield
    finally:
        _factor.reset(token)


def scoped_method(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        with support_scope(getattr(self, 'support_sigma', 2.)):
            return function(self, *args, **kwargs)
    return wrapped

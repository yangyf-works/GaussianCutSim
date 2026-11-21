"""The public GS data contract; CPU fitting records are not upload records."""
import numpy as np

CLIP_FIELDS = ('clip1_angle', 'clip1_split', 'clip2_angle', 'clip2_split')
FLOAT_FIELDS = ('positions', 'scale', 'rot', 'surface_color') + CLIP_FIELDS
GS_FIELDS = FLOAT_FIELDS + ('surface_id',)
REMOVED_FIELDS = ('f_dc', 'f_rest', 'opacity', 'surface_lighting', 'half_split', 'half_opacity',
                  'clip_planes', 'clip_lines', 'clip_count', 'keep_side')


def validate_lighting(value):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (3,) or not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError('Common lighting requires finite nonnegative ambient, diffuse, specular')
    return value


def validate_shininess(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError('Common shininess requires a finite positive exponent')
    return value


def validate_gaussian_attributes(attributes):
    for name in REMOVED_FIELDS:
        if name in attributes:
            raise ValueError(f'Unsupported GS attribute: {name}; allowed attributes: {", ".join(GS_FIELDS)}')
    unknown = attributes.keys() - set(GS_FIELDS)
    if unknown:
        raise ValueError(f'Unexpected GS attributes (CPU diagnostics must be resolved before upload): {sorted(unknown)}')
    positions = np.asarray(attributes.get('positions'))
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError('positions must have shape (N,3)')
    count = len(positions)
    shapes = dict(positions=(count, 3), scale=(count, 1), rot=(count, 4), surface_color=(count, 3))
    present = sum(name in attributes for name in CLIP_FIELDS)
    if present not in (0, 4):
        raise ValueError('Clipping requires all four angle/split attributes or none')
    if present:
        shapes.update({name: (count, 1) for name in CLIP_FIELDS})
    for name, shape in shapes.items():
        value = np.asarray(attributes.get(name))
        if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'{name} must be finite Float32 {shape}')
    if np.any(attributes['scale'] <= 0):
        raise ValueError('scale must be positive linear sigma')
    if not np.allclose(np.linalg.norm(attributes['rot'], axis=1), 1., rtol=0., atol=2e-6):
        raise ValueError('rot must contain normalized wxyz quaternions')
    color = attributes['surface_color']
    if np.any(color < 0) or np.any(color > 1):
        raise ValueError('surface_color must be in [0,1]')
    if 'surface_id' in attributes:
        value = np.asarray(attributes['surface_id'])
        if value.dtype != np.int32 or value.shape != (count, 1) or np.any(value < 0):
            raise ValueError('surface_id must be nonnegative Int32 (N,1)')
    for angle_name, split_name in zip(CLIP_FIELDS[::2], CLIP_FIELDS[1::2]):
        if angle_name in attributes:
            angle = attributes[angle_name]
            if np.any((angle != 4.) & ((angle < -np.float32(np.pi)) | (angle >= np.float32(np.pi)))):
                raise ValueError(f'{angle_name} must be in [-pi,pi) or the unused sentinel 4')
            if np.any((angle == 4.) & (attributes[split_name] != 0.)):
                raise ValueError('Unused clipping boundaries require angle=4, split=0')
    return attributes


def draw_from_calculation(records):
    """Lossless circular-scale projection of this application's CPU records.

    Coverage logits and boundary_split remain private to the preserved split
    fitter. Never reinterpret files or lit RGB here, nor average ellipse axes.
    """
    scale = np.asarray(records['scale'])
    if scale.ndim != 2 or scale.shape[1] not in (1, 3):
        raise ValueError('Calculation scale must have one or three components')
    if scale.shape[1] == 3:
        bad = np.flatnonzero(scale[:, 0] != scale[:, 1])
        if len(bad):
            raise ValueError(f'Anisotropic XY scales cannot be uploaded without changing shape: {len(bad)} rows, first {bad[:8].tolist()}')
    if 'coverage_logit' in records:
        expected = np.float32(np.log(.99 / .01))
        if np.any(records['coverage_logit'] != expected):
            raise ValueError('Variable alpha in the preserved split fitter is incompatible with fixed GS alpha 0.99')
    out = {key: value for key, value in records.items() if key in GS_FIELDS}
    out['scale'] = np.ascontiguousarray(scale[:, :1])
    validate_gaussian_attributes(out)
    return out

"""Camera-independent surface coverage contract for the native opaque shader.

gaussian_composite.comp uses 0.99*exp(-rho/2), rho<=9 and accumulated
transmittance < 0.5. Screen-space low-pass/occlusion are intentionally absent:
neighbouring geometry must cover the surface without help from the camera.
"""
import numpy as np

DEFAULT_DISPLAY_SCALE = 1.
# UI/audit multipliers use the 0.65 reference display size.
# Calculation sigma and finite material support use their own units.
REFERENCE_DISPLAY_SCALE = .65
PROFILE_SIGMA_SCALE = .75
REFERENCE_ALPHA = .99
OPAQUE_ALPHA_THRESHOLD = .5
OPAQUE_TRANSMITTANCE = 1. - OPAQUE_ALPHA_THRESHOLD
SURFEL_CUTOFF_SIGMA = 3.


def display_factor(sigma_scale):
    value = float(sigma_scale)
    if not np.isfinite(value) or value <= 0:
        raise ValueError('Invalid display scale')
    return value * REFERENCE_DISPLAY_SCALE / PROFILE_SIGMA_SCALE


def apply_display_scale(attributes, sigma_scale):
    """Apply the normalized UI multiplier; keep world clip boundaries fixed."""
    factor = display_factor(sigma_scale)
    attributes['scale'] *= factor
    for slot in (1, 2):
        attributes[f'clip{slot}_split'] /= factor


def opaque_geometry(snapshot, sigma_scale=DEFAULT_DISPLAY_SCALE):
    """Pack the published/uploaded sigma and two lines, including FP32 rounding."""
    from cutsim.rendering.profile_cut_clip import clipped_draw
    a = clipped_draw(snapshot, np.ones(len(snapshot.endpoints['positions']), bool))
    apply_display_scale(a, sigma_scale)
    sigma = a['scale'][:, 0].astype(float)
    lines = np.zeros((len(sigma), 7))
    for slot in (1, 2):
        angle = a[f'clip{slot}_angle'][:, 0]
        active = angle != 4.
        k = 3 * (slot - 1)
        lines[:, k] = np.where(active, np.cos(angle), 0.)
        lines[:, k+1] = np.where(active, np.sin(angle), 0.)
        lines[:, k+2] = np.where(active, -a[f'clip{slot}_split'][:, 0]*sigma, np.inf)
    lines[:, 6] = sigma
    # Empty certificates must not acquire opacity from the 3-sigma boundary.
    lines[snapshot.clipping.planes[:, 6] <= 0, 6] = 0.
    return lines

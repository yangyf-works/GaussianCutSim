"""Tool-local X-Z cutting section, shared by cache prewarming and inspection."""
import numpy as np


def ball_section_angles(radius, low, high, pitch, ball_angle_deg=None):
    """Shared meridian nodes for the strip section and profile-Dexel cells."""
    left = np.arccos(1.-low/radius)
    right = np.arccos(1.-min(high, radius)/radius)
    if ball_angle_deg is None:
        return np.linspace(left, right, max(1, int(np.ceil(radius*(right-left)/pitch)))+1)
    step = np.deg2rad(ball_angle_deg)
    regular = step*np.arange(int(np.ceil((np.pi/2.)/step))+1)
    interior = regular[(regular > left+1e-14) & (regular < right-1e-14)]
    return np.r_[left, interior, right]


def cutting_section_profile(kind, radius, low, high, pitch, ball_angle_deg=None):
    """Return (x, z, material-normal-x, material-normal-z, support) samples."""
    positions, normals, supports = [], [], []
    r, h = float(radius), float(pitch)
    def add(p, n, support=h):
        positions.extend(p); normals.extend(n); supports.extend([support]*len(p))
    if kind == 'ball' and low < r:
        angle = ball_section_angles(r,low,high,h,ball_angle_deg)
        support = h if ball_angle_deg is None else r*np.deg2rad(ball_angle_deg)
        for sign in (-1., 1.):
            x, z = sign*r*np.sin(angle), r*(1.-np.cos(angle))
            add(np.column_stack((x, z)), np.column_stack((-x/r, (r-z)/r)), support)
    side_low = max(low, r) if kind == 'ball' else low
    if high > side_low:
        z = np.linspace(side_low, high, max(1, int(np.ceil((high-side_low)/h)))+1)
        for sign in (-1., 1.):
            add(np.column_stack((np.full(len(z), sign*r), z)), np.tile([-sign, 0.], (len(z), 1)))
    for z, normal in ((low, [0., 1.]), (high, [0., -1.])):
        cap_r = np.sqrt(max(0., r*r-(z-r)**2)) if kind == 'ball' and z < r else r
        if cap_r == 0.:
            continue
        x = np.linspace(-cap_r, cap_r, max(1, int(np.ceil(2.*cap_r/h)))+1)
        add(np.column_stack((x, np.full(len(x), z))), np.tile(normal, (len(x), 1)))
    profile = np.column_stack((positions, normals, supports))
    profile.flags.writeable = False
    return profile

"""Direction-aware, prefix-stable labels for analytically continuous sweeps.

Only renderer identities change. Provenance, intervals and surface samples do
not. A direction tolerance accounts for Float64 normalization roundoff, not a
user-visible angle at which physically different paths should be smoothed.
"""
from dataclasses import dataclass
import numpy as np
from cutsim.core.commands import command_endpoints
from cutsim.core.tool_pose import tool_basis, DEFAULT_TOOL_AXIS


@dataclass(frozen=True)
class SurfaceJoin:
    motion: str
    reason: str
    previous_motion: int | None
    turn_degrees: float | None


@dataclass(frozen=True)
class ProfileContinuity:
    groups: np.ndarray
    decisions: tuple


def _motion(delta):
    length=float(np.linalg.norm(delta))
    if length==0.:return 'stationary',np.zeros(3)
    unit=delta/length
    if delta[0]==0. and delta[1]==0.:return ('axial_down' if delta[2]<0. else 'axial_up'),unit
    return ('horizontal' if delta[2]==0. else 'ramp'),unit


def analyze_profile_continuity(program):
    groups=np.arange(len(program),dtype=np.int64)
    endpoints=[command_endpoints(c) for c in program]
    motions=[_motion(b-a if c.tool_axis==DEFAULT_TOOL_AXIS else (b-a)@tool_basis(c.tool_axis))
             for c,(a,b) in zip(program,endpoints)]
    decisions=[];last_motion=None;reference=None
    tolerance=64*np.finfo(np.float64).eps
    for i,current in enumerate(program):
        motion,unit=motions[i];linked=False;join=False;angle=None;reason='first_move'
        if i:
            previous=program[i-1]
            compatible=(current.tool_type.lower() in ('flat','ball') and previous.tool_id==current.tool_id
                and previous.tool_type.lower()==current.tool_type.lower() and previous.radius==current.radius
                and previous.cutting_range==current.cutting_range and previous.tool_axis==current.tool_axis)
            linked=compatible and np.array_equal(endpoints[i-1][1],endpoints[i][0])
            reason='disconnected' if compatible else 'tool_or_profile_change'
        if not linked:last_motion=None;reference=None
        predecessor=last_motion
        if linked:
            if motion=='stationary':join=True;reason='stationary_at_joint'
            elif last_motion is None:join=True;reason='start_from_stationary'
            else:
                previous_motion,previous_unit=motions[last_motion]
                dot=float(previous_unit@unit);cross=float(np.linalg.norm(np.cross(previous_unit,unit)))
                angle=float(np.degrees(np.arctan2(cross,dot)))
                # Compare to the run's reference too: many tiny turns must not
                # accumulate into a bend while inheriting a single identity.
                straight=(float(reference@unit)>0. and np.linalg.norm(np.cross(reference,unit))<=tolerance
                          and dot>0. and cross<=tolerance)
                if straight:join=True;reason='same_forward_direction'
                elif previous_motion=='axial_down' and motion=='horizontal':
                    # A different motion direction, but the lower tool end fan
                    # and translated cross-section have a common tangent join.
                    join=True;reason='plunge_to_feed'
                elif dot<0. and cross<=tolerance:reason='reversal'
                elif motion=='ramp' or previous_motion=='ramp':reason='slope_or_heading_change'
                elif motion.startswith('axial') or previous_motion.startswith('axial'):reason='axial_transition'
                else:reason='turn'
        if join:groups[i]=groups[i-1]
        decisions.append(SurfaceJoin(motion,reason,predecessor,angle))
        if motion!='stationary':
            if not join or reason!='same_forward_direction':reference=unit
            last_motion=i
    groups.flags.writeable=False
    return ProfileContinuity(groups,tuple(decisions))


def continuous_profile_groups(program):
    return analyze_profile_continuity(program).groups

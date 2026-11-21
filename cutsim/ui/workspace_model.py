"""Presentation data and block-boundary computation jobs; no Open3D dependency."""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from cutsim.core.commands import command_endpoints


SHAPE_UPDATE_CHOICES = (('strip','Tool-profile Dexel (CUDA)'),)


def shape_update_label(mode):
    return dict(SHAPE_UPDATE_CHOICES).get(mode,mode)


@dataclass(frozen=True)
class ToolSummary:
    tool_id: int
    kind: str
    diameter: float
    blocks: tuple[int, ...]
    cutting_range: object = None
    gs_sampling: object = None

    @property
    def label(self):
        text = f'T{self.tool_id}  |  {self.kind.title()}  |  D{self.diameter:g} mm'
        if self.cutting_range is not None:
            text += f'  |  Cut {self.cutting_range.start:g}–{self.cutting_range.end:g} mm'
        if self.gs_sampling is not None and self.gs_sampling.mode == 'fixed':
            text += f'  |  GS {self.gs_sampling.pitch_mm:g} mm'
        if self.gs_sampling is not None and self.gs_sampling.mode == 'auto' and self.gs_sampling.diameter_divisions is not None:
            text += f'  |  GS D/{self.gs_sampling.diameter_divisions}'
        if self.kind == 'ball' and self.gs_sampling is not None and self.gs_sampling.ball_angle_deg is not None:
            text += f'  |  Ball {self.gs_sampling.ball_angle_deg:g}°'
        return text


def tools_in_program(program):
    """Group by tool ID, type, diameter, cutting range and sampling settings."""
    groups = {}
    for index, cmd in enumerate(program):
        key = (cmd.tool_id, cmd.tool_type.lower(), float(cmd.radius) * 2, cmd.cutting_range, cmd.gs_sampling)
        groups.setdefault(key, []).append(index)
    return [ToolSummary(*key[:3], tuple(blocks), key[3], key[4]) for key, blocks in groups.items()]


def path_length(cmd):
    if len(cmd.a) == 3:
        from math import dist
        return dist(cmd.a, cmd.b)
    return hypot(cmd.b[0] - cmd.a[0], cmd.b[1] - cmd.a[1])


def tool_details_html(tool, program):
    """Supplementary tool settings; geometry and block counts remain in the list."""
    rows = []
    sampling = tool.gs_sampling
    if sampling is not None:
        spacing = (f'{sampling.pitch_mm:g} mm' if sampling.mode == 'fixed' else
                   f'D / {sampling.diameter_divisions}' if sampling.diameter_divisions is not None else 'Auto')
        rows.append(('GS spacing', spacing))
        if tool.kind == 'ball' and sampling.ball_angle_deg is not None:
            rows.append(('Ball sampling angle', f'{sampling.ball_angle_deg:g}°'))
    return '<table width="100%" cellspacing="0" cellpadding="1">' + ''.join(
        f'<tr><td style="color:#a6b5c8">{label}</td><td align="right">{value}</td></tr>'
        for label, value in rows) + '</table>'


def next_tool_block(program, current):
    """First block after the current tool run; do not wrap at the final tool."""
    if not program or current >= len(program):
        return None
    if current < 0:
        return 0

    def key(cmd):
        # Tool orientation and machining stage may change without a tool change.
        return (cmd.tool_id, cmd.tool_type.lower(), cmd.radius, cmd.cutting_range, cmd.gs_sampling)

    tool = key(program[current])
    return next((i for i in range(current + 1, len(program)) if key(program[i]) != tool), None)


def playback_path_indices(program, current, project=None):
    """The contiguous operation using the current tool, respecting dataset stages."""
    if not 0 <= current < len(program):
        return range(0)

    def key(index):
        cmd = program[index]
        stage = None
        if project is not None and index < len(project.cutting_block_indices):
            stage = project.blocks[project.cutting_block_indices[index]].get('stage')
        return (cmd.tool_id, cmd.tool_type.lower(), cmd.radius, cmd.cutting_range, cmd.gs_sampling, stage, cmd.tool_axis)

    selected = key(current)
    start, end = current, current + 1
    while start > 0 and key(start - 1) == selected:
        start -= 1
    while end < len(program) and key(end) == selected:
        end += 1
    return range(start, end)


def block_label(index, cmd, computed):
    state = 'Ready' if index < computed else 'Pending'
    return f'{index + 1:03d}   T{cmd.tool_id} {cmd.tool_type.title()} D{2 * cmd.radius:g}   |   {state}'


def block_details(index, cmd, computed):
    cutting = cmd.cutting_range
    range_text = (f'\nCutting range from tip: {cutting.start:g}–{cutting.end:g} mm' if cutting is not None else '')
    if cmd.tool_axis != (0.,0.,1.):
        range_text += '\nTool axis: ' + ', '.join(f'{v:.4f}' for v in cmd.tool_axis)
    if len(cmd.a) == 3:
        a, b = command_endpoints(cmd)
        return (f'Block {index + 1:03d}  /  {"Computed" if index < computed else "Not computed"}\n'
                f'Tool: T{cmd.tool_id}  {cmd.tool_type.title()}  D{2 * cmd.radius:g} mm\n'
                f'Start XYZ:  {a[0]:.3f}, {a[1]:.3f}, {a[2]:.3f} mm\n'
                f'End XYZ:    {b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f} mm\n'
                f'Cut length:  {path_length(cmd):.3f} mm' + range_text)
    return (f'Block {index + 1:03d}  /  {"Computed" if index < computed else "Not computed"}\n'
            f'Tool: T{cmd.tool_id}  {cmd.tool_type.title()}  D{2 * cmd.radius:g} mm\n'
            f'Start XY:  {cmd.a[0]:.3f}, {cmd.a[1]:.3f} mm\n'
            f'End XY:    {cmd.b[0]:.3f}, {cmd.b[1]:.3f} mm\n'
            f'Tool-tip Z:  {cmd.depth:.3f} mm\n'
            f'Cut length:  {path_length(cmd):.3f} mm' + range_text)


def block_details_html(index, cmd, *, project=None, display=None):
    """Start/end coordinates and supplementary data for the selected command."""
    from html import escape
    source = {}
    source_index = None
    if project is not None and 0 <= index < len(project.cutting_block_indices):
        source_index = project.cutting_block_indices[index]
        source = project.blocks[source_index]
    a, b = command_endpoints(cmd)
    coordinates = '<tr><td></td>' + ''.join(
        f'<td align="right" style="color:#a6b5c8">{axis} ({unit})</td>'
        for axis, unit in zip('XYZAB', ('mm', 'mm', 'mm', '°', '°'))) + '</tr>'
    # Keep source angles unwrapped; tool-axis vectors do not determine A/B.
    for label, key, point in (('Start', 'start', a), ('End', 'end', b)):
        pose = source.get(key, ())
        angles = [f'{angle:.3f}' for angle in pose[3:]] if len(pose) == 5 else ['—', '—']
        values = [f'{value:.3f}' for value in point] + angles
        coordinates += f'<tr><td>{label}</td>' + ''.join(
            f'<td align="right">{value}</td>' for value in values) + '</tr>'
    rows = [('Move length', f'{path_length(cmd):.3f} mm')]
    if cmd.tool_axis != (0., 0., 1.) and not all(len(source.get(key, ())) == 5 for key in ('start', 'end')):
        rows.append(('Tool axis', ', '.join(f'{value:.4f}' for value in cmd.tool_axis)))
    if source_index is not None:
        if 'feed' in source:
            rows.append(('Feed', f'{source["feed"]:g} mm/min'))
    properties = ''.join(f'<tr><td style="color:#a6b5c8">{escape(label)}</td>'
                         f'<td align="right">{escape(value)}</td></tr>' for label, value in rows)
    description = escape(str(source['label'])) if source.get('label') else ''
    return (description + '<table width="100%" cellspacing="0" cellpadding="2">' + coordinates +
            '</table><table width="100%" cellspacing="0" cellpadding="2">' + properties + '</table>')


def block_timing_html(display, index):
    """Timing rows remain visible before measurements become available."""
    metrics = getattr(display, 'block_metrics', {}).get(index, {})
    history_keys = ('archive_seconds', 'read_seconds')
    history = (sum(metrics.get(key, 0.) for key in history_keys)
               if any(key in metrics for key in history_keys) else None)
    rows = [('Calculation', metrics.get('cutting_seconds')),
            ('Playback preparation', metrics.get('preparation_seconds')),
            ('History save/load', history)]
    contents = ('<tr><td style="color:#a6b5c8"><b>Process</b></td>'
                '<td align="right" style="color:#a6b5c8"><b>Time (ms)</b></td></tr>')
    for label, seconds in rows:
        value = '—' if seconds is None else f'{seconds*1000:.1f}'
        contents += f'<tr><td>{label}</td><td align="right">{value}</td></tr>'
    return '<table width="100%" cellspacing="0" cellpadding="2">' + contents + '</table>'


# Compatibility exports preserve existing callers and older replay pickle paths.
from cutsim.application.publication import ComputeEvent, CompletedBlock, DisplayState, display_material
from cutsim.application.jobs import ComputeJob


def block_timing_text(display, index):
    metrics=getattr(display,'block_metrics',{}).get(index)
    if not metrics:return ''
    cut=metrics.get('cutting_seconds',0.)
    prep=metrics.get('preparation_seconds',0.)
    history=metrics.get('archive_seconds',0.)+metrics.get('read_seconds',0.)
    return f'\nCalculation: {cut*1000:.1f} ms\nPlayback preparation: {prep*1000:.1f} ms\nHistory save/load: {history*1000:.1f} ms'

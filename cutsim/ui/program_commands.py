"""JSON instruction presentation, with an explicit mapping to cutting blocks."""
from dataclasses import dataclass
from html import escape

from cutsim.core.commands import command_endpoints


@dataclass(frozen=True)
class InstructionRow:
    values: tuple[str, ...]
    tool_id: int
    block_index: int | None = None
    source_index: int | None = None
    source: dict | None = None
    heading: bool = False


def pose_values(source, key):
    pose = list(source.get(key, ()))
    if len(pose) == 2:
        pose.append(source.get('depth'))
    pose += [None] * (5-len(pose))
    return tuple('—' if value is None else f'{value:.3f}' for value in pose[:5])


def instruction_rows(program, project=None):
    mapping = getattr(project, 'cutting_block_indices', None)
    if mapping is None:
        sources = [dict(kind='feed', start=command_endpoints(c)[0], end=command_endpoints(c)[1],
                        tool=f'T{c.tool_id}') for c in program]
        mapping = range(len(program))
    else:
        sources = project.blocks
    cuts = dict(zip(mapping, range(len(program))))
    tools = {t['id']: t for t in getattr(project, 'tools', ())}
    stages = getattr(project, 'stages', ())
    rows = []
    previous = None
    for line, source in enumerate(sources):
        block = cuts.get(line)
        command = program[block] if block is not None else None
        reference = source.get('tool', '')
        tool = tools.get(reference, {})
        tool_id = tool.get('tool_id', command.tool_id if command is not None else 0)
        stage = source.get('stage')
        group = (reference, tool_id, stage)
        if group != previous or source.get('kind') == 'toolchange':
            kind = tool.get('type', command.tool_type if command is not None else '')
            radius = tool.get('radius', command.radius if command is not None else None)
            parts = [f'T{tool_id}']
            if kind:
                parts.append(kind.title())
            if radius is not None:
                parts.append(f'Ø{2*radius:g} mm')
            if stage is not None:
                parts.append(stages[stage] if stage < len(stages) else f'Stage {stage+1}')
            rows.append(InstructionRow((' · '.join(parts),), tool_id, heading=True))
        previous = group
        motion = {'feed': 'Feed', 'rapid': 'Rapid', 'toolchange': 'Tool change'}.get(
            source.get('kind'), str(source.get('kind', '')))
        rows.append(InstructionRow((f'{line+1:03}', motion, *pose_values(source, 'end')),
                                   tool_id, block, line, source))
    return rows


def source_details_html(row):
    """Details for non-cutting commands, which have no simulation block."""
    source = row.source or {}
    table = '<tr><td></td>' + ''.join(f'<td align="right">{a}</td>' for a in
        ('X (mm)', 'Y (mm)', 'Z (mm)', 'A (°)', 'B (°)')) + '</tr>'
    for label, key in (('Start', 'start'), ('End', 'end')):
        table += f'<tr><td>{label}</td>' + ''.join(
            f'<td align="right">{value}</td>' for value in pose_values(source, key)) + '</tr>'
    note = escape(str(source.get('label', '')))
    feed = f'<br>Feed: {source["feed"]:g} mm/min' if 'feed' in source else ''
    return (note +
            '<table width="100%" cellspacing="0" cellpadding="2">' + table +
            '</table>' + feed + '<br><span style="color:#a6b5c8">Non-cutting instruction</span>')

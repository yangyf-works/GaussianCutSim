"""GaussianCutSim core / program implementation."""
from __future__ import annotations

from typing import List, Tuple
from cutsim.core.commands import CutCommand, command_endpoints


def load_program(path):
    """Load explicit XY/depth or XYZ straight tool-tip moves from JSON."""
    import json
    from pathlib import Path
    rows = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(rows, list):
        raise ValueError('A program must be a JSON array of cut commands')
    commands = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict): raise ValueError(f'Block {i+1} must be an object')
        try:
            row = dict(row, a=tuple(row['a']), b=tuple(row['b']))
            command = CutCommand(**row)
            if command.tool_type.lower() not in ('flat', 'ball'):
                raise ValueError('tool_type must be flat or ball')
            commands.append(command)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'Block {i+1}: {exc}') from exc
    return commands


def demo_program(program=None) -> List[CutCommand]:
    """Add a Z approach and retract to every machining pass, at clearance 5."""
    passes = list(default_program() if program is None else program)
    moves = []
    for cut in passes:
        start, end = command_endpoints(cut)
        approach = CutCommand((start[0], start[1], 5.), tuple(start), cut.radius,
                              tool_id=cut.tool_id, tool_type=cut.tool_type, cutting_range=cut.cutting_range,
                              gs_sampling=cut.gs_sampling, tool_axis=cut.tool_axis)
        retract = CutCommand(tuple(end), (end[0], end[1], 5.), cut.radius,
                             tool_id=cut.tool_id, tool_type=cut.tool_type, cutting_range=cut.cutting_range,
                             gs_sampling=cut.gs_sampling, tool_axis=cut.tool_axis)
        moves.extend((approach, cut, retract))
    return moves


def default_program() -> List[CutCommand]:
    """Compatibility entry point for the external 12-pass demo dataset."""
    from cutsim.core.machining_data import DEFAULT_PROJECT, load_project
    project = load_project(DEFAULT_PROJECT)
    return [command for command, index in zip(project.program, project.cutting_block_indices)
            if project.blocks[index].get('role') == 'cut']

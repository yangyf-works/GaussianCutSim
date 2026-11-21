"""Read machining datasets independently of the simulation and UI settings."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np

from cutsim.core.commands import CutCommand, CuttingRange
from cutsim.core.program import load_program
from cutsim.core.tool_sampling import ToolGSSampling
from cutsim.core.tool_pose import DEFAULT_TOOL_AXIS, normalized_axis, axis_from_ab


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / 'machining_data'
DEFAULT_PROJECT = DEFAULT_DATA_ROOT / 'mixed_demo' / 'project.json'


@dataclass(frozen=True)
class MachiningProject:
    path: Path
    name: str
    description: str
    stock_lower: tuple[float, ...]
    stock_upper: tuple[float, ...]
    program: tuple[CutCommand, ...]
    # Source blocks retain rapid moves, tool changes, labels and feeds.
    # Only feed blocks enter the cutting engine.
    blocks: tuple[dict, ...]
    cutting_block_indices: tuple[int, ...]
    tools: tuple[dict, ...]
    stages: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetEntry:
    path: Path
    name: str
    error: str = ''


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def stock_bounds(lower, upper):
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (lo.shape != (3,) or hi.shape != (3,) or not np.isfinite(lo).all()
            or not np.isfinite(hi).all() or np.any(hi <= lo)):
        raise ValueError('Stock lower/upper must be finite XYZ coordinates with lower < upper')
    return tuple(lo), tuple(hi)


def _project_commands(tool_rows, blocks, sampling_defaults=None):
    if not isinstance(tool_rows, list) or not isinstance(blocks, list):
        raise ValueError('Tools and toolpath files must each contain a JSON array')
    tools = {}
    tool_numbers = {}
    settings = {}
    for tool in tool_rows:
        reference = tool['id']
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError('Tool references must be nonempty strings')
        if reference in tools:
            raise ValueError(f'Duplicate tool reference: {reference}')
        tool_id = tool['tool_id']
        if not isinstance(tool_id, int) or isinstance(tool_id, bool) or not 0 <= tool_id <= 2147483647:
            raise ValueError('Tool IDs must be nonnegative 32-bit integers')
        if tool_id in tool_numbers:
            raise ValueError(f'Duplicate tool number T{tool_id}: {tool_numbers[tool_id]} and {reference}; '
                             'each tool in a project must have a unique tool_id')
        tool_numbers[tool_id] = reference
        if tool['type'] not in ('flat', 'ball'):
            raise ValueError(f'Unsupported tool type: {tool["type"]}')
        if tool['type'] == 'ball' and not tool.get('cut_cylinder', True):
            raise ValueError('Ball tools without cylindrical cutting are not supported')
        if not np.isfinite(tool['radius']) or tool['radius'] <= 0:
            raise ValueError('Tool radius must be finite and positive')
        if 'cutting_range' not in tool:
            raise ValueError(f'Tool {reference}: cutting_range is required')
        CuttingRange.parse(tool['cutting_range'])
        settings[reference] = ToolGSSampling.parse(tool.get('gs_sampling'), sampling_defaults)
        if 'profile_dexel' in tool:
            raise ValueError('Use the shared gs_sampling settings instead of profile_dexel')
        tools[reference] = tool
    commands, indices = [], []
    for index, block in enumerate(blocks):
        try:
            if any(key in block for key in ('radius', 'tool_type', 'flute_length', 'cut_cylinder', 'cutting_range', 'gs_sampling', 'profile_dexel')):
                raise ValueError('Tool geometry belongs in tools.json; toolpath blocks reference a tool')
            tool = tools[block['tool']]
            kind = block['kind']
            if kind not in ('feed', 'rapid', 'toolchange'):
                raise ValueError(f'Unsupported motion kind: {kind}')
            if 'feed' in block and (not np.isfinite(block['feed']) or block['feed'] <= 0):
                raise ValueError('Feed must be finite and positive')
            if 'stage' in block and (not isinstance(block['stage'], int) or block['stage'] < 0):
                raise ValueError('Stage must be a nonnegative integer')
            endpoints = [np.asarray(block[key], dtype=float) for key in ('start', 'end')]
            if endpoints[0].shape != endpoints[1].shape:
                raise ValueError('Start and end must use the same coordinate format')
            for pose in endpoints:
                if pose.shape not in ((2,), (3,), (5,)) or not np.isfinite(pose).all():
                    raise ValueError('A pose must contain finite XY, XYZ or XYZAB values')
            axis = normalized_axis(block.get('tool_axis', DEFAULT_TOOL_AXIS))
            if len(endpoints[0]) == 5:
                if 'tool_axis' in block:
                    raise ValueError('Use either XYZAB poses or tool_axis, not both')
                start_axis = axis_from_ab(*endpoints[0][3:])
                if kind == 'feed' and not np.allclose(endpoints[0][3:], endpoints[1][3:], atol=1.e-12, rtol=0.):
                    raise ValueError('Indexed cutting requires a fixed tool axis; simultaneous 5-axis is not supported')
                axis = start_axis
            if kind == 'feed':
                commands.append(CutCommand(tuple(endpoints[0][:3]), tuple(endpoints[1][:3]),
                    tool['radius'], depth=block.get('depth'),
                    tool_id=tool['tool_id'], tool_type=tool['type'], cutting_range=tool['cutting_range'],
                    gs_sampling=settings[block['tool']], tool_axis=axis))
                indices.append(index)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'Source block {index + 1}: {exc}') from exc
    return tuple(commands), tuple(indices)


def load_project(path) -> MachiningProject:
    """Load project.json (or its directory); never write into a dataset."""
    path = Path(path).resolve()
    if path.is_dir():
        path = path / 'project.json'
    try:
        data = _read_json(path)
        if data.get('schema_version') != 2:
            raise ValueError('Unsupported machining dataset schema_version (expected 2: project, tools, toolpath)')
        if data.get('units') != 'mm':
            raise ValueError('Machining dataset units must be mm')
        name = data['name']
        if not isinstance(name, str) or not name.strip():
            raise ValueError('Machining dataset name must be nonempty')
        lo, hi = stock_bounds(data['stock']['lower'], data['stock']['upper'])
        tool_path = (path.parent / data['tools']).resolve()
        motion_path = (path.parent / data['toolpath']).resolve()
        if any(not p.is_relative_to(path.parent) for p in (tool_path, motion_path)):
            raise ValueError('Tools and toolpath files must be inside their machining dataset directory')
        if len({path, tool_path, motion_path}) != 3:
            raise ValueError('Project, tools and toolpath must use three separate files')
        tools, blocks = _read_json(tool_path), _read_json(motion_path)
        sampling_defaults = ToolGSSampling()
        if isinstance(tools, dict):
            if set(tools)-{'defaults', 'tools'}:
                raise ValueError('Tools file accepts only defaults and tools')
            defaults = tools.get('defaults', {})
            if not isinstance(defaults, dict) or set(defaults)-{'gs_sampling'}:
                raise ValueError('Tool defaults accept only gs_sampling (shared by strip and profile Dexel)')
            sampling_defaults = ToolGSSampling.parse(defaults.get('gs_sampling'))
            tools = tools['tools']
        commands, indices = _project_commands(tools, blocks, sampling_defaults)
        if not commands:
            raise ValueError('A machining dataset must contain at least one cutting block')
        stages = data.get('stages', [])
        if not isinstance(stages, list) or not all(isinstance(stage, str) for stage in stages):
            raise ValueError('Stages must be an array of names')
        return MachiningProject(path, name, str(data.get('description', '')), lo, hi,
                                commands, tuple(blocks), indices, tuple(tools), tuple(stages))
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'{path.name}: {exc}') from exc


def discover_datasets(root=DEFAULT_DATA_ROOT, *, validate=True):
    """List projects; menus may defer full tool/path validation until loading."""
    entries = []
    for path in sorted(Path(root).glob('*/project.json')):
        name = path.parent.name
        try:
            data = _read_json(path)
            name = str(data.get('name', name))
            if validate:
                project = load_project(path)
                entries.append(DatasetEntry(project.path, project.name))
            else:
                entries.append(DatasetEntry(path.resolve(), name))
        except (OSError, ValueError, AttributeError) as exc:
            entries.append(DatasetEntry(path.resolve(), name, str(exc)))
    return entries


def simulation_input(*, project_path=None, program_path=None, data_root=DEFAULT_DATA_ROOT,
                     shape_update='strip', ball_only=False):
    """Shared CLI/UI input selection; dataset stock follows the selected program."""
    if shape_update != 'strip':
        raise ValueError('The current CUDA Tool-profile engine requires strip compatibility mode')
    project = None
    if project_path is not None:
        project = load_project(project_path)
    elif program_path is None:
        path = Path(data_root) / 'mixed_demo' / 'project.json'
        if not path.is_file():
            path = next((entry.path for entry in discover_datasets(data_root) if not entry.error), None)
            if path is None:
                raise ValueError(f'No compatible machining datasets in {data_root}')
        project = load_project(path)
    if project is not None:
        commands = list(project.program)
    elif program_path is not None:
        commands = load_program(program_path)
    else:
        from cutsim.core.program import default_program
        commands = default_program()
    if ball_only:
        indices = [i for i, cmd in enumerate(commands) if cmd.tool_type.lower() == 'ball']
        commands = [commands[i] for i in indices]
        if not commands:
            raise ValueError('The selected machining data contains no ball-tool cutting blocks')
        if project is not None:
            project = replace(project, name=project.name + ' (Ball only)', program=tuple(commands),
                cutting_block_indices=tuple(project.cutting_block_indices[i] for i in indices))
    return commands, project

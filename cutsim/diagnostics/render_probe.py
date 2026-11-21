"""Save actual GS uploads and replay render-only variants in isolated workers.

No cutting, resampling, boundary fitting, or visible window is used in a replay.
The frozen camera matrices are checked after reconstruction before rendering.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from cutsim.rendering.clipping import CLIP_ATTRIBUTE_NAMES


from cutsim.rendering.gs_schema import GS_FIELDS, validate_gaussian_attributes, validate_lighting, validate_shininess

UPLOAD_FIELDS = GS_FIELDS
VARIANTS = ('baseline', 'no_clip', 'larger_capacity', 'reverse_order', 'no_postprocess',
            'unlit', 'white_background')
CAPACITY_FIELDS = ('gaussian_splat_max_tiles_per_splat', 'gaussian_splat_avg_tiles_per_splat',
                   'gaussian_splat_max_tile_entries_total')


def array_hash(arrays):
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def save_case(path, attributes, camera, width, height, material, *, context=None):
    validate_gaussian_attributes(attributes)
    arrays = {name: np.array(attributes[name], copy=True) for name in UPLOAD_FIELDS if name in attributes}
    cam = dict(model=np.asarray(camera.get_model_matrix()).tolist(),
               view=np.asarray(camera.get_view_matrix()).tolist(),
               projection=np.asarray(camera.get_projection_matrix()).tolist(),
               near=float(camera.get_near()), far=float(camera.get_far()),
               fov=float(camera.get_field_of_view()),
               fov_type=str(camera.get_field_of_view_type()).split('.')[-1])
    meta = dict(format=2, width=int(width), height=int(height), camera=cam,
                input_hash=array_hash(arrays), context=context or {},
                material={name: int(getattr(material, name)) for name in CAPACITY_FIELDS},
                lighting=validate_lighting(getattr(material, 'gaussian_splat_surface_lighting', [1., 0., 0.])).tolist(),
                shininess=validate_shininess(getattr(material, 'gaussian_splat_surface_shininess', 64.)),
                visible=bool((context or {}).get("gs_visible",True)),
                opaque_surface=bool(getattr(material,'gaussian_splat_opaque_surface',False)),
                background=[.055, .065, .08, 1.],
                scope='work GS only; tool meshes, debug layers and markers excluded')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays, metadata=np.array(json.dumps(meta)))
    return meta


def load_case(path):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata']))
        if meta.get('format') != 2:
            raise ValueError('Render case format 2 with unlit surface colors is required')
        arrays = {name: archive[name].copy() for name in archive.files if name != 'metadata'}
    if array_hash(arrays) != meta['input_hash']:
        raise ValueError('Frozen render case format/hash mismatch')
    validate_gaussian_attributes(arrays)
    validate_lighting(meta['lighting'])
    validate_shininess(meta.get('shininess', 64.))
    return arrays, meta


def variant_attributes(attributes, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    out = {key: value.copy() for key, value in attributes.items()}
    if variant == 'no_clip':
        count = len(out['positions'])
        for name in CLIP_ATTRIBUTE_NAMES:
            out[name] = np.full((count, 1), 4. if name.endswith('angle') else 0., np.float32)
    elif variant == 'reverse_order':
        out = {key: value[::-1].copy() for key, value in out.items()}
    return out


def restore_camera(camera, meta, rendering):
    saved = meta['camera']; model = np.asarray(saved['model'])
    eye = model[:3, 3]
    camera.look_at(eye - model[:3, 2], eye, model[:3, 1])
    fov_type = getattr(rendering.Camera.FovType, saved['fov_type'])
    camera.set_projection(saved['fov'], meta['width'] / meta['height'],
                          saved['near'], saved['far'], fov_type)
    errors = {}
    for name in ('model', 'view', 'projection'):
        actual = np.asarray(getattr(camera, 'get_' + name + '_matrix')())
        expected = np.asarray(saved[name])
        errors[name] = float(np.max(np.abs(actual - expected)))
        if not np.allclose(actual, expected, rtol=2e-6, atol=2e-5):
            raise ValueError(f'Camera reconstruction differs: {name}, max error {errors[name]}')
    return errors


def render_worker(path, variant, output, surface_mode='saved'):
    import open3d as o3d
    from open3d.visualization import rendering
    from cutsim.rendering.open3d_renderer import gaussian_tensor_cloud, configure_gaussian_material

    attrs, meta = load_case(path)
    upload = variant_attributes(attrs, variant)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    renderer = rendering.OffscreenRenderer(meta['width'], meta['height'])
    scene = renderer.scene
    background = [1., 1., 1., 1.] if variant == 'white_background' else meta['background']
    scene.set_background(np.asarray(background, np.float32))
    scene.view.set_shadowing(False)
    if variant == 'no_postprocess':
        scene.view.set_post_processing(False)
        scene.view.set_antialiasing(False)
        scene.view.set_ambient_occlusion(False)
    material = rendering.MaterialRecord(); material.shader = 'gaussianSplat'
    opaque=bool(meta.get('opaque_surface',False)) if surface_mode=='saved' else surface_mode=='opaque'
    configure_gaussian_material(material, [1.,0.,0.] if variant=='unlit' else meta['lighting'], opaque,
                                meta.get('shininess', 64.))
    for name, value in meta['material'].items():
        setattr(material, name, value)
    if variant == 'larger_capacity':
        tiles = ((meta['width'] + 15) // 16) * ((meta['height'] + 15) // 16)
        material.gaussian_splat_max_tiles_per_splat = max(tiles, material.gaussian_splat_max_tiles_per_splat)
        target = max(32 * 1024 * 1024, material.gaussian_splat_max_tile_entries_total)
        material.gaussian_splat_avg_tiles_per_splat = max(material.gaussian_splat_avg_tiles_per_splat,
                                                          min(tiles, target // max(1, len(upload['positions']))))
        material.gaussian_splat_max_tile_entries_total = target
    if len(upload['positions']):
        cloud = gaussian_tensor_cloud(o3d, upload)
        scene.add_geometry('work', cloud, material)
        scene.show_geometry('work', meta['visible'])
    errors = restore_camera(scene.camera, meta, rendering)
    started = time.perf_counter()
    # Native rendering is isolated in this child; save all evidence before exit.
    image = renderer.render_to_image()
    depth = np.asarray(renderer.render_to_depth_image(True)).copy()
    if not o3d.io.write_image(str(output / f'{variant}.png'), image):
        raise OSError('Could not save render comparison image')
    np.save(output / f'{variant}_depth.npy', depth)
    report = dict(variant=variant, input_hash=meta['input_hash'], upload_hash=array_hash(upload),
                  camera=meta['camera'], camera_errors=errors, count=len(upload['positions']),
                  open3d_version=o3d.__version__, seconds=time.perf_counter() - started,
                  background=background, lighting=list(map(float,getattr(material, 'gaussian_splat_surface_lighting', [1., 0., 0.]))),
                  opaque_surface=bool(getattr(material, 'gaussian_splat_opaque_surface', False)),
                  visible=meta['visible'],
                  material={name: int(getattr(material, name)) for name in CAPACITY_FIELDS},
                  finite_depth_pixels=int(np.isfinite(depth).sum()),
                  image_sha256=hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),
                  scope=meta['scope'], worker_exit='explicit exit after writing artifacts; native teardown not tested')
    (output / f'{variant}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)
    # Avoid the independently known native renderer-destruction crash. This
    # only applies to this disposable diagnostic worker, never the application.
    sys.stdout.flush(); sys.stderr.flush()
    import os
    os._exit(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('--variant', choices=VARIANTS, default='baseline')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--surface-mode', choices=('saved','transparent','opaque'), default='saved',
                        help='Override the saved mode without changing GS or camera')
    parser.add_argument('--timeout', type=float, default=240.)
    args = parser.parse_args()
    output = args.output or args.case.with_suffix('')
    if args.worker:
        render_worker(args.case, args.variant, output,args.surface_mode)
    else:
        output.mkdir(parents=True, exist_ok=True)
        variants = VARIANTS if args.all else (args.variant,)
        failed = False
        for variant in variants:
            command = [sys.executable, '-m', 'cutsim.diagnostics.render_probe', str(args.case.resolve()), '--worker',
                       '--variant', variant, '--output', str(output.resolve()),'--surface-mode',args.surface_mode]
            try:
                run = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
                log = run.stdout + '\n' + run.stderr
                code = run.returncode
            except subprocess.TimeoutExpired as exc:
                log = f'Worker timed out after {args.timeout}s\n{exc}'
                code = -1
            (output / f'{variant}.log').write_text(log, encoding='utf-8')
            print(f'{variant}: exit={code}; results={output}', flush=True)
            failed |= code != 0
        if failed:
            raise SystemExit(1)


if __name__ == '__main__':
    main()

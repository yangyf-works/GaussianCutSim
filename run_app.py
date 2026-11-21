"""Start the CUDA Tool-profile Dexel operator workspace."""
import argparse
import os
import sys

from cutsim.diagnostics.profiling import install_performance_logging


def require_cuda():
    from cutsim.cuda.profile_dexel_cuda import runtime
    try:
        runtime()
    except Exception as exc:
        raise RuntimeError(f'CUDA is required for Tool-profile Dexel: {exc}') from exc


def main():
    parser = argparse.ArgumentParser(description='GaussianCutSim')
    from cutsim._version import __version__
    parser.add_argument('--version', action='version', version='GaussianCutSim '+__version__)
    parser.add_argument('--open3d-ui', action='store_true', help='Open the Open3D operator workspace')
    parser.add_argument('--initial-splat', type=float, default=.75, help='Initial stock sampling size in mm')
    parser.add_argument('--gs-density', type=float, default=1., help='GS density multiplier: 0.10-8.0; default 1.0 (100%%)')
    parser.add_argument('--ball-only', action='store_true', help='Use only ball-tool blocks from the example program')
    data = parser.add_mutually_exclusive_group()
    data.add_argument('--program', help='JSON command array (uses the default stock)')
    data.add_argument('--project', help='Machining dataset directory or project.json (stock, tools, program)')
    from cutsim.core.machining_data import DEFAULT_DATA_ROOT, simulation_input
    parser.add_argument('--data-dir', default=str(DEFAULT_DATA_ROOT), help='Machining data library shown in the UI')
    args = parser.parse_args()
    try:
        from cutsim.core.cuda_cache import configure_cuda_cache
        configure_cuda_cache()
    except (ValueError,RuntimeError) as exc:
        parser.error(str(exc))
    if not args.initial_splat > 0 or not args.initial_splat < float('inf'):
        parser.error('--initial-splat must be a finite positive number')
    from cutsim.geometry.sampling_density import validate_gs_density
    try:
        validate_gs_density(args.gs_density)
    except ValueError as exc:
        parser.error(str(exc))
    from cutsim.application.process_job import ProcessComputeJob
    job = ProcessComputeJob()
    job.start_background_warmup()
    try:
        try:
            require_cuda()
        except RuntimeError as exc:
            parser.error(str(exc))
        from cutsim.cutting.realtime_simulator import SweepDrivenRealtimeSimulator
        install_performance_logging(detailed=os.environ.get('CUTSIM_PROFILE', '0') == '1')
        try:
            program, project = simulation_input(project_path=args.project, program_path=args.program,
                data_root=args.data_dir, shape_update='strip', ball_only=args.ball_only)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        stock = dict(stock_lo=project.stock_lower, stock_hi=project.stock_upper) if project else {}
        sim = SweepDrivenRealtimeSimulator(initial_splat=args.initial_splat, shape_update_mode='strip',
                                          program=program, gs_density=args.gs_density, **stock)
        sim.machining_project = project
        if args.open3d_ui:
            from cutsim.ui.workspace_gui import MachiningWorkspaceGUI
            app = MachiningWorkspaceGUI(sim, compute_job=job)
        else:
            try:
                from cutsim.ui.qt_workspace import QtMachiningApplication
            except ModuleNotFoundError as exc:
                if exc.name != 'PySide6':
                    raise
                parser.error('PySide6 is required. Install requirements.txt with this Python, or use --open3d-ui.')
            app = QtMachiningApplication(sim, data_root=args.data_dir, compute_job=job)
        app.run()
    finally:
        job.shutdown()



if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()

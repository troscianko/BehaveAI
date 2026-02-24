import sys
import os
import click
from pathlib import Path


def _default_projects_dir() -> Path:
    exe = Path(sys.executable)
    # Pixi environment: find .pixi in path parts
    try:
        pixi_idx = exe.parts.index(".pixi")
        return Path(*exe.parts[:pixi_idx]) / "behaveai_projects"
    except ValueError:
        pass
    # Standard venv: check for pyvenv.cfg two levels up
    venv_root = exe.parent.parent
    if (venv_root / "pyvenv.cfg").exists():
        return venv_root.parent / "behaveai_projects"
    # Global install fallback
    return Path.home() / "BehaveAI" / "projects"


DEFAULT_PROJECTS_DIR = _default_projects_dir()


@click.group(invoke_without_command=True)
@click.pass_context
@click.option("--project", type=click.Path(exists=True, file_okay=False), default=None,
              help="Path to an existing BehaveAI project directory.")
@click.option("--projects-dir", type=click.Path(file_okay=False), default=None,
              help="Override the default projects directory.")
def cli(ctx, project, projects_dir):
    """BehaveAI - animal tracking and behaviour classification."""
    if ctx.invoked_subcommand is None:
        from behaveai.BehaveAI import launch
        launch(
            project_path=project,
            projects_dir=Path(projects_dir) if projects_dir else DEFAULT_PROJECTS_DIR,
        )


@cli.command()
@click.argument("settings", type=click.Path(exists=True, dir_okay=False))
@click.option("--input",  "input_dir",  type=click.Path(exists=True, file_okay=False), default=None,
              help="Override the input directory from the settings file.")
@click.option("--output", "output_dir", type=click.Path(file_okay=False), default=None,
              help="Override the output directory from the settings file.")
def run(settings, input_dir, output_dir):
    """Run headless batch processing using a settings INI file."""
    from behaveai.classify_track import run_batch
    run_batch(config_path=settings, input_dir=input_dir, output_dir=output_dir)


@cli.command()
@click.argument("input",  type=click.Path(exists=True))
@click.argument("output", type=click.Path(), required=False, default=None)
@click.option("--strategy", type=click.Choice(["exponential", "sequential"]), default="exponential", show_default=True,
              help="Frame accumulation strategy.")
@click.option("--exp-a",    type=float, default=0.5, show_default=True,
              help="Exponential decay for the green (medium-term) channel.")
@click.option("--exp-b",    type=float, default=0.8, show_default=True,
              help="Exponential decay for the red (older) channel.")
@click.option("--lum-weight", type=float, default=0.7, show_default=True,
              help="Blend weight of original luminance in the output (0-1).")
@click.option("--rgb-multipliers", type=str, default="4.0,4.0,4.0", show_default=True,
              help="Comma-separated scaling factors for the R,G,B motion channels.")
@click.option("--chromatic-tail-only", is_flag=True, default=False,
              help="Show only the chromatic motion tail, suppressing base luminance from motion channels.")
@click.option("--scale-factor", type=float, default=1.0, show_default=True,
              help="Resize factor applied to each frame before processing.")
@click.option("--frame-skip", type=int, default=0, show_default=True,
              help="Number of frames to skip between processed frames.")
@click.option("--motion-threshold", type=int, default=0, show_default=True,
              help="Brightness offset applied to output (negative darkens low-motion areas).")
@click.option("--compress", is_flag=True, default=False,
              help="Re-encode the output with FFmpeg H.264 after writing (requires ffmpeg in PATH).")
@click.option("--crf", type=int, default=23, show_default=True,
              help="H.264 quality for --compress (lower = better quality, 18-28 is typical).")
def motion(input, output, strategy, exp_a, exp_b, lum_weight, rgb_multipliers,
           chromatic_tail_only, scale_factor, frame_skip, motion_threshold, compress, crf):
    """Convert a video (or folder of videos) to a motion-enhanced output.

    INPUT can be a single video file or a directory. If a directory is given,
    all video files within it are processed and written to OUTPUT as a directory.

    The output encodes motion history chromatically: blue = recent movement,
    green = medium-term, red = older movement.

    \b
    Examples:
      behaveai motion input.mp4
      behaveai motion input.mp4 output.mp4
      behaveai motion input.mp4 output.mp4 --strategy exponential --exp-a 0.5
      behaveai motion videos/ --chromatic-tail-only
      behaveai motion videos/ motion_videos/
    """
    from behaveai.motion import process_motion_batch

    # Derive default output path if not given
    if output is None:
        p = Path(input)
        if p.is_dir():
            output = str(p.parent / (p.name + "_motion"))
        else:
            output = str(p.parent / (p.stem + "_motion" + p.suffix))
        click.echo(f"Output: {output}")

    try:
        multipliers = tuple(float(x) for x in rgb_multipliers.split(","))
        if len(multipliers) != 3:
            raise ValueError
    except ValueError:
        raise click.BadParameter("Must be three comma-separated floats, e.g. '4.0,4.0,4.0'",
                                 param_hint="--rgb-multipliers")

    process_motion_batch(
        input_path=input,
        output_path=output,
        strategy=strategy,
        exp_a=exp_a,
        exp_b=exp_b,
        lum_weight=lum_weight,
        rgb_multipliers=multipliers,
        chromatic_tail_only=chromatic_tail_only,
        scale_factor=scale_factor,
        frame_skip=frame_skip,
        motion_threshold=motion_threshold,
        compress=compress,
        crf=crf,
    )
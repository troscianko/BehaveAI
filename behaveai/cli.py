import click
import sys
from pathlib import Path

def _default_projects_dir() -> Path:
    exe = Path(sys.executable)
    venv_root = exe.parent.parent

    # Standard venv
    if (venv_root / "pyvenv.cfg").exists():
        return venv_root.parent / "behaveai_projects"

    # Pixi: <project>/.pixi/envs/<env_name>/bin/python
    try:
        pixi_idx = exe.parts.index(".pixi")
        project_root = Path(*exe.parts[:pixi_idx])
        return project_root / "behaveai_projects"
    except ValueError:
        pass

    return Path.home() / "BehaveAI" / "projects"

DEFAULT_PROJECTS_DIR = _default_projects_dir()


@click.group(invoke_without_command=True)
@click.pass_context
@click.option(
    "--project",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Path to a project directory to pre-select in the launcher.",
)
def cli(ctx, project):
    """BehaveAI — animal tracking and behaviour classification."""
    if ctx.invoked_subcommand is None:
        from behaveai.launcher import launch
        launch(project_path=project, projects_dir=DEFAULT_PROJECTS_DIR)


# @cli.command()
# @click.argument("settings", type=click.Path(exists=True, dir_okay=False))
# @click.option(
#     "--input", "input_dir",
#     type=click.Path(exists=True, file_okay=False),
#     default=None,
#     help="Override the input directory from the settings file.",
# )
# @click.option(
#     "--output", "output_dir",
#     type=click.Path(file_okay=False),
#     default=None,
#     help="Override the output directory from the settings file.",
# )
# def run(settings, input_dir, output_dir):
#     """Run batch video processing headlessly against a settings file."""
#     from behaveai.classify_track import process_video
#     process_video(settings_path=settings, input_dir=input_dir, output_dir=output_dir)
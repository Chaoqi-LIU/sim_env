if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)


import importlib
import click

COMMANDS = {
    "rlbench":   "policy_research.env.rlbench.gen_data:gen_rlbench_data",
    "drake":     "policy_research.env.drake.gen_data:gen_drake_data",
    "metaworld": "policy_research.env.metaworld.gen_data:gen_metaworld_data",
}

COMMANDS_HELP = {
    "rlbench":   "Generate RLBench data",
    "drake":     "Generate Drake data",
    "metaworld": "Generate MetaWorld data",
}

def _load_click_command(spec: str) -> click.Command:
    """Import and return a Click command object from 'module:attr' spec."""
    try:
        module_path, attr_name = spec.split(":")
    except ValueError as e:
        raise click.ClickException(f"Invalid command spec '{spec}'. Expected 'module:attr'") from e

    try:
        mod = importlib.import_module(module_path)
    except Exception as e:
        raise click.ClickException(
            f"Failed to import module '{module_path}' for '{spec}'. Error: {e}"
        ) from e

    try:
        cmd = getattr(mod, attr_name)
    except AttributeError as e:
        raise click.ClickException(
            f"Module '{module_path}' does not define '{attr_name}'"
        ) from e

    # Ensure it’s a Click command (Click 8/9 safe)
    if not isinstance(cmd, click.Command):
        raise click.ClickException(
            f"Loaded '{spec}', but it is not a Click command "
            "(missing @click.command() / @click.group() decorator?)."
        )
    return cmd

class LazyGroup(click.Group):
    """A Group that lazy-loads subcommands on demand."""
    def list_commands(self, ctx):
        return sorted(COMMANDS.keys())

    def get_command(self, ctx, name):
        spec = COMMANDS.get(name)
        if spec is None:
            return None
        return _load_click_command(spec)

    def format_commands(self, ctx, formatter):
        """Show subcommands and short help without importing modules."""
        rows = []
        for name in self.list_commands(ctx):
            help_text = COMMANDS_HELP.get(name, "")
            rows.append((name, help_text))
        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)

@click.command(cls=LazyGroup)
def gen_data():
    """Data generation CLI (lazy import)."""
    pass


if __name__ == "__main__":
    gen_data()
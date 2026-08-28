import click
import spin

@click.command()
@click.option(
    '--fix',
    is_flag=True,
    default=False,
    required=False,
)
def lint(fix):
    """🔦 Run lint and typing checks

    """
    ruff_flags = ["--fix"] if fix else []
    spin.util.run(["ruff", "check", "numpy_financial/", "benchmarks/"] + ruff_flags)

    spin.util.run(["pyright"])
    spin.util.run(["mypy", "--no-incremental", "."])

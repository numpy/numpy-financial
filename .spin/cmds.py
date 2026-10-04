import os
import sys

import click
import spin
from spin.cmds import meson


@click.command()
@click.option('--tests', '-t', multiple=True,
              help='Select benchmarks by name or regular expression.')
@meson.build_dir_option
@click.pass_context
def bench(ctx, tests, build_dir):
    """Run benchmarks against the local build without saving results."""
    args = [sys.executable, '-m', 'asv', 'run', '--python=same',
            '--dry-run', '--show-stderr']
    for test in tests:
        args.extend(['--bench', test])
    ctx.invoke(meson.build, build_dir=build_dir)
    meson._set_pythonpath(build_dir)
    env = os.environ.copy()
    # ASV removes PYTHONPATH unless it is explicitly passed this way.
    env['ASV_PYTHONPATH'] = env.get('PYTHONPATH', '')
    spin.util.run(args, env=env)


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

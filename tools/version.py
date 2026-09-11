#!/usr/bin/env python3

"""Determine and print the version number.

Used in the top level ``meson.build``.
"""

from pathlib import Path

INIT = Path(__file__).parent.parent / "numpy_financial" / "__init__.py"


def version_from_init():
    """Extract the version string from ``numpy_financial/__init__.py``."""
    lines = INIT.read_text().splitlines()
    version_line = next(line for line in lines if line.startswith("__version__ ="))
    return version_line.split(" = ")[1].strip().strip("\"'")


if __name__ == "__main__":
    print(version_from_init())

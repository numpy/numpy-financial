#!/usr/bin/env python


def get_version():
    """
    Find the value assigned to __version__ in numpy_financial/__init__.py.

    This function assumes that there is a line of the form

        __version__ = "version-string"

    in that file.  It returns the string version-string, or None if such a
    line is not found.
    """
    with open("numpy_financial/__init__.py", "r") as f:
        for line in f:
            s = [w.strip() for w in line.split("=", 1)]
            if len(s) == 2 and s[0] == "__version__":
                return s[1][1:-1]


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.add_subpackage('numpy_financial')
    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(name='numpy-financial',
          version=get_version(),
          configuration=configuration)

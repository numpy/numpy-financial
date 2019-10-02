#!/usr/bin/env python


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.add_subpackage('numpy_financial')
    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(name='numpy-financial',
          version='0.0.1.dev0',
          configuration=configuration)
